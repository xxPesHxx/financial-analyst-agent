import logging
import concurrent.futures
from typing import Callable, Any, Dict, Type, Literal
from pydantic import BaseModel, ValidationError, Field
from app.guardrails import SecurityError
from app import tools

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Argument Schemas
class StockArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")

class AnalysisArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    horizon: Literal['1d', '1w', '1m', '1y'] = Field(..., description="Time horizon (e.g. '1d', '1w')")

class RSIArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, func: Callable, schema: Type[BaseModel]):
        self._tools[name] = {"func": func, "schema": schema}

    def get_tool_schema(self, name: str):
        return self._tools.get(name, {}).get("schema")

    def list_tools(self):
        return list(self._tools.keys())

    def dispatch(self, tool_name: str, args: dict, timeout: float = 5.0) -> Any:
        logger.info(f"Tool call: {tool_name} with args {args}")
        if tool_name not in self._tools:
            logger.error(f"Tool {tool_name} not found")
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_def = self._tools[tool_name]
        func = tool_def["func"]
        schema = tool_def["schema"]
        
        # Validate args
        try:
            validated_args = schema(**args)
        except ValidationError as e:
            logger.error(f"Validation error for {tool_name}: {e}")
            raise ValueError(f"Validation error: {e}")
            
        # Execute with timeout
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, **validated_args.model_dump()) 
                result = future.result(timeout=timeout)
                logger.info(f"Tool {tool_name} success")
                return result
        except concurrent.futures.TimeoutError:
            logger.error(f"Tool {tool_name} timed out")
            raise TimeoutError(f"Tool {tool_name} timed out")
        except SecurityError as e:
            logger.warning(f"Security error in {tool_name}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Execution error in {tool_name}: {e}")
            # The prompt says handle validation errors gracefully (no stack traces).
            # The API level should catch this and return a nice message.
            raise RuntimeError(f"Tool execution failed: {str(e)}")

# Global Registry
registry = ToolRegistry()
registry.register("get_stock_price", tools.get_stock_price, StockArgs)
registry.register("technical_analysis_forecast", tools.technical_analysis_forecast, AnalysisArgs)
registry.register("get_rsi", tools.get_rsi, RSIArgs)
