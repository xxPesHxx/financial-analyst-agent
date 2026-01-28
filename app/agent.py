import os
import json
import logging
from typing import Dict, Any, List
from app.registry import registry
from app.rag import rag_engine
from app.guardrails import detect_injection, validate_output, SecurityError
from app.schemas import ToolCallSchema, AskResponse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        is_empty = not self.api_key
        is_template = self.api_key and self.api_key.startswith("sk-...") or self.api_key == "sk-..."
        
        self.mock_mode = is_empty or is_template
        
        if self.mock_mode:
            reason = "Key is empty" if is_empty else "Key matches template 'sk-...'"
            logger.warning(f"Agent starting in MOCK mode. Reason: {reason}")
            print(f"DEBUG: Mock Mode ON. Reason: {reason}")
        else:
            masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key else "None"
            logger.info(f"Agent starting in LIVE mode. Key: {masked_key}")
            print(f"DEBUG: Live Mode ON. Key: {masked_key}")

    def run(self, query: str) -> AskResponse:

        if detect_injection(query):
            logger.warning(f"Prompt injection detected: {query}")
            raise SecurityError("Prompt injection detected")

        context_docs = rag_engine.retrieve(query, top_k=2)
        context_str = "\n".join([f"[{d['meta']['source']}#{d['meta']['chunk_id']}] {d['content']}" for d in context_docs])
        
        system_prompt = f"""Jesteś asystentem do analizy danych finansowych.
        WAŻNE: Nie jesteś doradcą finansowym. Twoje odpowiedzi służą wyłącznie celom edukacyjnym i informacyjnym.
        Nigdy nie rekomenduj zakupu ani sprzedaży akcji.
        Opieraj się wyłącznie na faktach dostarczonych przez narzędzia i kontekst RAG.
        Unikaj języka sugerującego pewność zysku.
        
        Używaj dostępnych narzędzi, aby odpowiadać na pytania o ceny akcji, analizę techniczną i RSI.
        
        Relevant Financial News:
        {context_str}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response_msg = self._call_llm(messages, tools=self._get_tool_definitions())
        
        tool_calls_result = []
        answer = ""

        if response_msg.get("tool_calls"):
            tool_call = response_msg["tool_calls"][0] 
            tool_name = tool_call["function"]["name"]
            
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}
                logger.error("Failed to parse tool arguments JSON")

            try:
                result = registry.dispatch(tool_name, arguments)
                tool_output = json.dumps(result)
            except Exception as e:
                tool_output = f"Error: {str(e)}"
                logger.error(f"Tool execution error: {e}")
        
            tool_calls_result.append(ToolCallSchema(tool=tool_name, args=arguments))
            
            messages.append({"role": "assistant", "content": response_msg.get("content"), "tool_calls": response_msg.get("tool_calls")})
            messages.append({
                "role": "tool", 
                "tool_call_id": tool_call["id"], 
                "name": tool_name, 
                "content": tool_output
            })
            
            final_response = self._call_llm(messages)
            answer = final_response["content"]
        else:
            answer = response_msg["content"]

        if not answer:
            answer = "I executed the tool but could not generate a response."

        try:
            validate_output(answer)
        except SecurityError as e:
            answer = "Response blocked by security guardrails."

        return AskResponse(answer=answer, tool_calls=tool_calls_result)

    def _get_tool_definitions(self):
        definitions = []
        for name in registry.list_tools():
            schema = registry.get_tool_schema(name)
            if not schema:
                continue
            json_schema = schema.model_json_schema()
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Function to {name}",
                    "parameters": json_schema
                }
            })
        return definitions

    def _call_llm(self, messages, tools=None) -> Dict[str, Any]:
        if self.mock_mode:
            return self._mock_llm(messages, tools)
        
        if self.api_key.startswith("AIza"):
            return self._call_gemini(messages, tools)
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0
            )
            message = completion.choices[0].message
            
            tool_calls_list = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls_list.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return {
                "content": message.content,
                "tool_calls": tool_calls_list if tool_calls_list else None
            }
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return {"content": "Sorry, I am unable to connect to the AI service at the moment.", "tool_calls": None}

    def _call_gemini(self, messages, tools=None) -> Dict[str, Any]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            gemini_tools = []
            if tools:
                pass 

            model_name = 'gemini-2.5-flash' 
            model = genai.GenerativeModel('gemini-2.5-flash') 
            
            chat_history = []
            system_instruction = ""
            last_user_msg = ""

            for m in messages:
                if m["role"] == "system":
                    system_instruction += m["content"] + "\n"
                elif m["role"] == "user":
                    last_user_msg = m["content"]
                    chat_history.append({"role": "user", "parts": [m["content"]]})
                elif m["role"] == "assistant":
                    chat_history.append({"role": "model", "parts": [m["content"] or ""]})
                elif m["role"] == "tool":
                    chat_history.append({"role": "user", "parts": [f"Tool Context/Result: {m['content']}"]})

            
            if tools:
                tool_descriptions = json.dumps(tools, indent=2)
                system_instruction += f"\n\nYOU HAVE ACCESS TO THESE TOOLS:\n{tool_descriptions}\n\n"
                system_instruction += "To call a tool, verify you must use it. If yes, output ONLY valid JSON using this schema: {\"tool_calls\": [{\"id\": \"call_...\", \"function\": {\"name\": \"...\", \"arguments\": \"{...}\"}}]}\n"
                system_instruction += "If no tool is needed, just answer normally."

            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash', 
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
            chat = model.start_chat(history=chat_history[:-1] if len(chat_history) > 0 else [])
            response = chat.send_message(last_user_msg)
            
            text = response.text
            
            try:
                import re
                json_match = re.search(r'\{.*"tool_calls".*\}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    return {"content": None, "tool_calls": data["tool_calls"]}
            except:
                pass

            return {"content": text, "tool_calls": None}

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return {"content": f"Sorry, Gemini API error: {str(e)}", "tool_calls": None}

    def _mock_llm(self, messages, tools):
        last_msg = messages[-1]
        
        if last_msg["role"] == "user":
            content = last_msg["content"].lower()
            
            if tools:
                matched_tool = None
                args = {}
                
                from app.guardrails import validate_ticker, SecurityError
                ticker = None
                
                words = last_msg["content"].split()
                for w in words:
                    clean_w = w.strip(".,!?\"'")
                    try:
                        validate_ticker(clean_w)
                        ticker = clean_w
                        break
                    except SecurityError:
                        continue
                
                if not ticker and ("../../" in content or "drop table" in content):
                     return {"content": "Blocked: Invalid ticker format or malicious input detected."}

                if not ticker:
                    ticker = "AAPL"
                
                if ("price" in content or "cena" in content or "kurs" in content) and "get_stock_price" in registry.list_tools():
                    matched_tool = "get_stock_price"
                    args = {"ticker": ticker}
                elif ("forecast" in content or "analy" in content or "prognoza" in content) and "technical_analysis_forecast" in registry.list_tools():
                    matched_tool = "technical_analysis_forecast"
                    args = {"ticker": ticker, "horizon": "1w"}
                elif "rsi" in content and "get_rsi" in registry.list_tools():
                    matched_tool = "get_rsi"
                    args = {"ticker": ticker}

                if matched_tool:
                    return {
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_mock_{matched_tool}",
                            "function": {
                                "name": matched_tool,
                                "arguments": json.dumps(args)
                            }
                        }]
                    }
        
            system_instruction = next((m["content"] for m in messages if m["role"] == "system"), "")
            system_instruction = next((m["content"] for m in messages if m["role"] == "system"), "")
            news_header = "Relevant Financial News:"
            if news_header in system_instruction:
                try:
                    news_content = system_instruction.split(news_header)[1].strip()
                    if len(news_content) > 10:
                        return {"content": f"Based on the retrieved financial news (Mock Mode): \n\n{news_content}\n\n(This is a direct dump/summary of context because real LLM synthesis is disabled)"}
                except:
                    pass

            return {"content": "I am a Mock Agent. I didn't understand. Try: 'Cena AAPL', 'Analiza TSLA', 'RSI BTC'."}

        if last_msg["role"] == "tool":
            tool_res = last_msg["content"]
            try:
                res_dict = json.loads(tool_res)
                return {"content": f"The result is: {res_dict}. (Mock Answer)"}
            except:
                return {"content": f"Tool returned: {tool_res}. (Mock Answer)"}
            
        return {"content": "I'm not sure what to do."}
