# prompt Chinese

src_session_prompt_zh = "\"query\": {query}. \"items\": {items}.\n"
src_reason_prompt_zh = """
以下是用户的搜索历史 {session_prompt}, 其中每条记录包含用户的查询(query)以及该查询下用户点击的物品(items)。"""
src_reason_instruction_zh = """
请你分析用户搜索历史中的查询和点击的内容，总结用户的兴趣主题、关注点、风格倾向或偏好类型，并以如下 JSON 格式输出：
{
  "用户偏好总结": "请详细的描述用户可能的偏好，例如感兴趣的领域、常搜内容类型、点击偏好方向等。"
}
请仅基于给出的搜索历史信息进行分析，不要引入外部知识或主观猜测。如信息不足，可模糊表达。"""

rec_reason_prompt_zh = """这是用户的推荐历史 {rec_his}，其中每条记录表示用户点击过的一个物品。"""
rec_reason_instruction_zh = """
请你根据提供的用户推荐历史信息，总结该用户可能的兴趣偏好、风格倾向和物品喜好类型，并以如下 JSON 格式输出：
{
  "用户偏好总结": "在这里详细的描述用户的偏好，例如兴趣主题、常见类型、风格倾向等。"
}
只能根据提供的历史进行分析，不要引入外部信息或进行无根据的猜测。如果信息不足，可以适当模糊表述。"""

# prompt English

src_session_prompt_en = "\"query\": {query}. \"items\": {items}.\n"
src_reason_prompt_en = """
Here is the user's search history {session_prompt}, where each record contains the user's query and the items the user clicked on under that query."""
src_reason_instruction_en = """
Please analyze the queries and clicked items in the user's search history, and summarize the user's interest topics, areas of focus, style tendencies, or preference types. 
Output your analysis in the following JSON format: 
{ 
  "User Preference Summary": "Please provide a detailed description of the user's possible preferences, such as areas of interest, commonly searched content types, and clicking preference directions." 
} 
Please base your analysis solely on the provided search history information, without introducing external knowledge or subjective assumptions. If the information is insufficient, you may express the analysis in a generalized manner."""

rec_reason_prompt_en = """Here is the user's recommendation history {rec_his}, where each record represents an item the user has clicked on."""
rec_reason_instruction_en = """
Please analyze the provided user recommendation history and summarize the user's possible interests, style tendencies, and preferred item types. 
Output your analysis in the following JSON format: 
{ 
  "User Preference Summary": "Provide a detailed description of the user's preferences here, such as interest topics, common item types, and style tendencies." 
} 
You must base your analysis solely on the given history without introducing external information or making unfounded assumptions. If the available information is insufficient, you may express the summary in a generalized manner."""
