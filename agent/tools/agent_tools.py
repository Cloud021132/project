import os
import json
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

default_user_location = "深圳"
default_user_id = "1001"
default_month = "2025-10"
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"]
external_data = {}
city_weather_map = {
    "深圳": "深圳天气为晴天，气温28摄氏度，空气湿度58%，南风2级，AQI22，整体舒适。",
    "合肥": "合肥天气为多云，气温25摄氏度，空气湿度65%，东南风3级，AQI46，适合外出。",
    "杭州": "杭州天气为小雨转阴，气温24摄氏度，空气湿度70%，东北风2级，AQI36，稍微潮湿。",
    "北京": "北京天气为晴朗，气温27摄氏度，空气湿度42%，西北风3级，AQI33，空气较好。",
    "上海": "上海天气为阵雨，气温26摄氏度，空气湿度76%，东风2级，AQI52，外出记得带伞。",
    "广州": "广州天气为雷阵雨，气温30摄氏度，空气湿度78%，南风4级，AQI58，注意防雷。",
    "成都": "成都天气为阴天，气温22摄氏度，空气湿度62%，东南风2级，AQI40，温度宜人。",
    "武汉": "武汉天气为小雨，气温23摄氏度，空气湿度68%，北风2级，AQI49，湿度偏高。",
    "西安": "西安天气为晴天，气温29摄氏度，空气湿度35%，西南风3级，AQI28，适合户外活动。",
    "重庆": "重庆天气为多云，气温27摄氏度，空气湿度60%，南风3级，AQI44，气温适中。",
}

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市的模拟天气概览,返回该城市的具体天气状态、温度、湿度、风向和 AQI。")
def get_weather(city: str) -> str:
    if not city or not city.strip():
        return "请提供城市名称，以便查询天气。"

    normalized = city.strip()
    if normalized in city_weather_map:
        return city_weather_map[normalized]

    return f"暂未收录{city}的天气数据，当前仅支持深圳、合肥、杭州、北京、上海、广州、成都、武汉、西安、重庆等城市。"

@tool(description="获取用户所在城市的名称，以纯字符串形式返回。默认返回当前示例用户所在城市：深圳。")
def get_user_location(city: str | None = None) -> str:
    return (city or default_user_location).strip() or default_user_location

@tool(description="获取用户的id，以纯字符串形式返回。默认返回示例用户 ID：1001。")
def get_user_id(user_id: str | None = None) -> str:
    return (user_id or default_user_id).strip() or default_user_id

@tool(description="获取当前月份，以纯字符串形式返回。默认返回示例月份：2025-06。")
def get_current_month(month: str | None = None) -> str:
    return (month or default_month).strip() or default_month

def generator_external_data():
    """
    {
        "user_id":{
            "month":{"特征":xxx"效率"：xxx,......}
            "month":{"特征":xxx"效率"：xxx,......}
            "month":{"特征":xxx"效率"：xxx,......}
        }
        "user_id":{
            "month":{"特征":xxx"效率"：xxx,......}
            "month":{"特征":xxx"效率"：xxx,......}
            "month":{"特征":xxx"效率"：xxx,......}
        }
        "user_id":{
            "month":{"特征":xxx"效率"：xxx,......}
            "month":{"特征":xxx"效率"：xxx,......}
            "month":{"特征":xxx"效率"：xxx,......}
        }
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].strip().strip('"')
                feature: str = arr[1].strip().strip('"')
                efficiency: str = arr[2].strip().strip('"')
                consumables: str = arr[3].strip().strip('"')
                comparison: str = arr[4].strip().strip('"')
                time: str = arr[5].strip().strip('"')

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征":feature,
                    "效率":efficiency,
                    "耗材":consumables,
                    "对比":comparison,
                }




@tool(description="从外部系统中获取用户的使用记录，以纯字符串形式返回，如果未检测到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generator_external_data()

    try:
        data = external_data[user_id][month]
        return json.dumps(data, ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


if __name__ == '__main__':
    print(fetch_external_data.func("1002", "2025-01"))