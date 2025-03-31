# ---------- 用户端执行入口 main.py ----------

import asyncio
from config import Config
from model_client import BeiXiaoChuanClient, CLIPClient, HierarchicalModelClient
from agents import BaseAgent, MarketingAgent, OperationAgent, ResearchAgent
from tools.registry import ToolRegistry
from cognition.memory import MemoryPool
from monitoring.reporter import MetricsReporter
import sys
import os
import uuid
import logging

logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.getcwd())


class Application:
    def __init__(self):
        """初始化核心组件"""
        self.config = Config()
        self.llm_client = self._init_llm_client()
        self.clip_client = self._init_clip_client()
        self.tool_registry = self._init_tool_registry()
        self.agents = self._init_agents()
        self.memory = MemoryPool(embedding_dim=self.config.embed_dim)
        self.metrics_reporter = self._init_metrics()

    def _init_tool_registry(self) -> ToolRegistry:
        """初始化工具注册中心"""
        registry = ToolRegistry(dependencies={
            "config": self.config,
            "clip_client": self.clip_client,
            "db_engine": self._init_async_db(),
            "model_client": HierarchicalModelClient(self.config),
            "llm": self.llm_client
        })

        from tools.rag_tool import RAGTool
        from tools.nl2sql_tool import NL2SQLTool
        from tools.forecast_tool import ForecastTool
        from tools.report_tool import ReportTool
        
        RAGTool.register(registry)
        NL2SQLTool.register(registry)
        ForecastTool.register(registry)
        ReportTool.register(registry)
        
        return registry   

    def _init_async_db(self):
        """初始化异步数据库连接"""
        from sqlalchemy.ext.asyncio import create_async_engine
        return create_async_engine(
            self.config.postgresql_uri,
            pool_size=self.config.max_async_connections,
            echo=True  # 调试时开启SQL日志
        )

    def _init_llm_client(self) -> BeiXiaoChuanClient:
        """初始化大模型客户端"""
        return BeiXiaoChuanClient(self.config)

    def _init_clip_client(self) -> CLIPClient:
        """初始化多模态编码客户端"""
        return CLIPClient(self.config)

    def _init_agents(self) -> dict[str, BaseAgent]:
        """初始化Agent集群"""
        return {
            "marketing": MarketingAgent(self.config, self.tool_registry),
            "operation": OperationAgent(self.config, self.tool_registry),
            "research": ResearchAgent(self.config, self.tool_registry)
        }

    def _init_metrics(self) -> MetricsReporter:
        """初始化监控报告"""
        reporter = MetricsReporter(interval=60)
        asyncio.create_task(reporter.start())
        return reporter

    async def _generate_embedding(self, request: dict) -> None[list[float]]:
        """生成多模态嵌入"""
        try:
            if image_url := request.get("image_url"):
                return await self.clip_client.encode_image_from_url(image_url)
            elif text := request.get("text"):
                return await self.clip_client.encode_text(text)
            return None
        except Exception as e:
            print(f"嵌入生成失败: {str(e)}")
            return None

    async def process_request(self, request: dict) -> dict:
        """处理用户请求的全流程"""
        try:
            # 1. 生成上下文嵌入
            embedding = await self._generate_embedding(request)

            # 2. 记忆检索
            context = {
                "query": request,
                "memory": self.memory.retrieve(embedding) if embedding else []
            }

            # 3. 路由到对应Agent
            agent_type = request.get("type", "marketing")
            selected_agent = self.agents.get(agent_type, self.agents["marketing"])


            # 4. 执行Agent工作流
            result = await selected_agent.arun(input={
                "query": request["text"],
                "intermediate_steps": []
            })

            # 5. 更新记忆池
            if embedding is not None and result:
                self.memory.store(
                    embedding,
                    {"query": request["text"], "result": result}
                )

            # 6. 记录路由指标
            expected_type = agent_type.lower()
            actual_type = type(selected_agent).__name__.lower()
            self.metrics_reporter.record_routing(expected_type == actual_type)

            return result
        except KeyError as e:
            return {"error": f"无效的Agent类型: {str(e)}"}
        except Exception as e:
            logger.exception("请求处理严重错误")
            return {
                "error": "系统繁忙，请稍后重试",
                "code": 500,
                "request_id": uuid.uuid4().hex
            }


# 测试用例
if __name__ == "__main__":
    """主程序入口"""
    app = Application()

    test_cases = [
        "设计美的集团智能空调的618促销方案，需要包含竞品分析",
        "生成2014年Q2的资产负债表趋势分析报告",
        "评估新能源汽车充电桩市场的投资机会"
    ]

    asyncio.run((lambda: asyncio.gather(*[
        app.process_request({"type": "marketing", "text": query}).add_done_callback(
            lambda fut: (
                print(f"Query: {query}"),
                print(f"Result: {fut.result()}"),
                print("=" * 50)
            )
        ) for query in test_cases
    ]))())
