# vLLM Strict Mode 实现分析:0.18.0 vs 0.24.0

> 分析基于本地代码树 `vllm-releases-v0.18.0` 与 `vllm-0.24.0` 的逐文件对比。

## 一、Strict Mode 的设计目标(来自 0.24.0 文档)

| `tool_choice` | Schema 约束 | 行为 |
| --- | --- | --- |
| Named function | 始终启用(structural tag) | arguments 保证符合函数参数 schema |
| `"required"` | 始终启用(structural tag) | 同上,且模型必须产生至少一个 tool call |
| `"auto"` | **仅当至少一个 tool 设 `strict: true`** | 有 strict → structural tag 约束;无 strict → 自由生成 + 文本提取 |
| `"none"` | N/A | 不产生 tool call |

全局开关 `VLLM_ENFORCE_STRICT_TOOL_CALLING`(默认 `true`)控制 structural-tag 路径的总开关;设为 `false` 则不附加 structural tag,但不影响 named/required 的 schema-derived 约束。

---

## 二、0.18.0 的现状:完全不存在 Strict Mode

### 2.1 缺失项清单(全局搜索 0 结果)

| 概念 | 0.18.0 |
| --- | --- |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` 环境变量 | ❌ 不存在 |
| `structural_tag_model` 类属性 | ❌ 不存在 |
| `get_structural_tag` 方法 | ❌ 不存在 |
| `supports_required_and_named` 类属性 | ❌ 不存在 |
| `structural_tag_registry.py` 文件 | ❌ 不存在 |
| `__init_subclass__` 钩子 | ❌ 不存在 |
| `AnthropicTool.strict` 字段 | ❌ 不存在 |
| ToolParser 构造函数接收 `tools` | ❌ 只有 `__init__(tokenizer)` |

### 2.2 0.18.0 的 tool calling 约束路径

```
serving层: tool_parser(tokenizer).adjust_request(request)
                                    │
                    ┌───────────────▼────────────────┐
                    │ ToolParser.adjust_request      │  (abstract_tool_parser.py:56)
                    │   get_json_schema_from_tools()  │
                    │   ├─ named/required → 设 structured_outputs.json
                    │   └─ auto → 返回 None(无约束)
                    └────────────────────────────────┘
```

- **named / required**:通过 `get_json_schema_from_tools`(`utils.py:191`)生成 JSON schema,写入 `request.structured_outputs.json`,由 structured outputs 后端在生成期约束。这条路径 **0.18.0 早就有**,与 structural tag 无关。
- **auto**:`get_json_schema_from_tools` 对 `auto` 返回 `None` → 无任何生成期约束 → 模型自由生成 → `extract_tool_calls` 从原始文本正则提取。

### 2.3 0.18.0 的提取路径(无 supports_required_and_named 区分)

`engine/serving.py:1104` 的 `_parse_tool_calls_from_content`(ChatCompletions)与 `abstract_parser.py:388` 的 `_parse_tool_calls`(Responses):

```python
if isinstance(tool_choice, ToolChoiceFunction):        # named (Responses)
    ...直接取 content 作为 arguments
elif isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):  # named (Chat)
    ...直接取 content 作为 arguments
elif tool_choice == "required":                         # required
    ...TypeAdapter(list[FunctionDefinition]).validate_json(content)
elif tool_parser_cls and enable_auto_tools and (auto/None):  # auto
    tool_parser_cls(tokenizer).extract_tool_calls(content, request)
```

三条路径**彼此独立**,没有"required/named 降级为 auto 式提取"的概念。

---

## 三、0.24.0 Strict Mode 的五层架构

### 3.1 第一层:全局环境变量

`vllm/envs.py:207` 与 `envs.py:1582`:

```python
VLLM_ENFORCE_STRICT_TOOL_CALLING: bool = True

environment_variables = {
    ...
    "VLLM_ENFORCE_STRICT_TOOL_CALLING": lambda: (
        os.getenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "True").lower() in ("true", "1")
    ),
}
```

- 默认 `True`,即默认启用 structural-tag 约束。
- **仅影响 structural-tag 路径**,不影响 named/required 的 JSON-schema 路径(文档明确)。

### 3.2 第二层:ToolParser 基类新增类属性 + `__init_subclass__`

`abstract_tool_parser.py:59-71`:

```python
class ToolParser:
    supports_required_and_named: bool = True
    structural_tag_model: str | None = None
    engine_based_streaming: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (cls.structural_tag_model is not None
                and envs.VLLM_ENFORCE_STRICT_TOOL_CALLING):
            cls.supports_required_and_named = False
```

**关键机制**:当一个 parser 子类声明了 `structural_tag_model` 且全局开关为 `True` 时,`__init_subclass__` 自动把 `supports_required_and_named` 置为 `False`。这意味着:

- 该 parser 的 required/named **不再走老的 JSON 提取路径**;
- 而是统一交给 structural tag 在**生成期**约束输出格式,提取时按 auto 式解析即可。

构造函数也变了(`abstract_tool_parser.py:73`):`__init__(tokenizer, tools=None)`,把 tools 存到 `self.tools`。

### 3.3 第三层:`get_structural_tag` + `structural_tag_registry.py`(核心判断)

`abstract_tool_parser.py:168-185`:

```python
def get_structural_tag(self, request, *, reasoning=False):
    if self.structural_tag_model is None:        # parser 不支持
        return None
    if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:  # 全局开关关闭
        return None
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
    return get_model_structural_tag(
        model=self.structural_tag_model,
        tools=request.tools, tool_choice=request.tool_choice, reasoning=reasoning)
```

`structural_tag_registry.py`(0.24.0 全新文件)的判断逻辑:

```python
def _any_tool_strict(tools):
    """任一 tool 的 strict == True 即返回 True"""
    for tool in tools:
        if isinstance(tool, FunctionTool) and tool.strict is True:
            return True
        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is True:
            return True
    return False

def get_model_structural_tag(model, tools, tool_choice, reasoning):
    if not tools or tool_choice == "none":
        return None
    # ★ auto 模式的 opt-in 关键:无 strict 则不约束
    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
    # required / named:跳过 strict 检查,总是生成 tag
    dumped_tools = [_dump_tool_for_xgrammar(t) for t in tools]
    ...
    return get_xgrammar_model_structural_tag(model=model, tools=..., tool_choice=..., reasoning=...)
```

**`tool_choice` 三分支语义在此实现**:

| tool_choice | 行为 |
| --- | --- |
| `none` | 返回 `None`(不约束) |
| `auto` | **必须** `_any_tool_strict` 为真才生成 tag;否则 `None` → 自由生成 |
| `required` / named | **跳过 strict 检查**,总是生成 tag |

`_get_function_parameters`(`structural_tag_registry.py:207`)处理 per-tool 级别:

```python
def _get_function_parameters(function):
    if getattr(function, "strict", None) is False:
        return True              # strict=False → 放松,arguments 无 schema 约束
    return function.parameters if function.parameters is not None else True
```

**支持的模型注册表**:

```python
XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({
    "llama", "kimi", "deepseek_r1", "deepseek_v3_1", "qwen_3_5",
    "qwen_3_coder", "qwen_3", "harmony", "deepseek_v3_2", "glm_4_7", "deepseek_v4",
})
VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"hermes"})  # 0.24.0 新增 minimax 走 vllm 注册
```

xgrammar 内置模型直接调 `get_xgrammar_model_structural_tag`;vllm 自有的(hermes、minimax)通过 `@register_vllm_structural_tag` 装饰器注册的 builder 构建。

### 3.4 第四层:DelegatingParser 的 `_apply_structural_tag` 注入

`abstract_parser.py:455-502`(0.18.0 无此方法):

```python
def adjust_request(self, request):
    if self._reasoning_parser:
        request = self._reasoning_parser.adjust_request(request)
    if self._tool_parser:
        request = self._apply_structural_tag(request)   # ★ 新增:先打 structural tag
    if self._tool_parser:
        request = self._tool_parser.adjust_request(request)
    return request

def _apply_structural_tag(self, request):
    if (self._tool_parser.structural_tag_model is None or not request.tools):
        return request
    need = (tool_choice in {"auto", "required"} or isinstance(named))
    if not need:
        return request
    tag = self._tool_parser.get_structural_tag(request, reasoning=False)
    if tag is None:
        return request
    request.structured_outputs = StructuredOutputsParams(
        structural_tag=json.dumps(tag.model_dump()))
    request.response_format = None   # Responses: request.text = None
    return request
```

随后 `ToolParser.adjust_request`(`abstract_tool_parser.py:130-135`)**短路**:

```python
def adjust_request(self, request):
    if not request.tools:
        return request
    structured_outputs = getattr(request, "structured_outputs", None)
    if (structured_outputs is not None
            and structured_outputs.structural_tag is not None):
        return request        # ★ 已用 structural tag,跳过 JSON-schema 路径
    # 否则走原来的 get_json_schema_from_tools → structured_outputs.json
    ...
```

**两条路径互斥**:structural tag 设置后不再设 `structured_outputs.json`。

### 3.5 第五层:提取期 `supports_required_and_named` 控制路径

`abstract_parser.py:388-453`(非流式)与 `:604-663`(流式):

```python
supports_required_and_named = tool_parser.supports_required_and_named
is_named = isinstance(tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam))
is_required = tool_choice == "required"
is_auto = enable_auto_tools and (
    tool_choice == "auto" or tool_choice is None
    or (not supports_required_and_named and (is_named or is_required))  # ★ 降级
)

if is_named and supports_required_and_named:
    ...标准 named 提取
elif is_required and supports_required_and_named:
    ...标准 required JSON 提取
elif is_auto:
    ...extract_tool_calls(auto 式)   # ← structural tag parser 的 required/named 走这里
```

对于声明了 `structural_tag_model` 的 parser,`supports_required_and_named=False`,因此 named/required **降级**走 auto 式 `extract_tool_calls`——因为生成阶段已被 structural tag 约束成正确格式,直接解析即可。

---

## 四、三个 API surface 的 `strict` 字段贯通

| API | 字段位置 | 0.18.0 | 0.24.0 |
| --- | --- | --- | --- |
| Chat Completion | `ChatCompletionToolsParam.function.strict` | openai 协议已有 | 已有 |
| Responses | `FunctionTool.strict` | openai SDK 已有 | 已有 |
| Anthropic Messages | `AnthropicTool.strict` | ❌ 无 | ✅ `protocol.py:78` 新增 |

Anthropic 的转换(`serving.py:562-590`):

```python
def _convert_tools(cls, anthropic_request, req):
    for tool in anthropic_request.tools:
        tools.append(ChatCompletionToolsParam.model_validate({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": tool.strict,          # ★ 0.24.0 新增透传
                "defer_loading": tool.defer_loading,
            },
        }))
```

转换后统一进入 ChatCompletion 协议,后续 `_any_tool_strict` 检查自然覆盖 Anthropic。

---

## 五、`StructuredOutputsParams.structural_tag` 字段

该字段 **0.18.0 已存在**(用于 reasoning parser 如 gptoss、以及用户直接传 `response_format.type == "structural_tag"`)。但 0.18.0 的 **tool calling 从未使用**它。

0.24.0 把它接入 tool calling 链路:

```
_apply_structural_tag 写入 structured_outputs.structural_tag
    → request.py:96 识别为 STRUCTURAL_OUTPUT_OPTIONS.STRUCTURAL_TAG
    → backend_xgrammar.py:345-361 用 xgr.Grammar.from_structural_tag 编译成受限解码语法
```

---

## 六、OpenAI strict-schema 风格的落地方式

文档建议:`additionalProperties: false` / 全字段 required / `["string","null"]` 表可选。

vLLM **不强制改写**用户 schema,而是**透传**:`_get_function_parameters` 把 `function.parameters` 原样塞进 `JSONSchemaFormat(json_schema=...)`(见 `_hermes_tool_tags` / `_minimax_tool_tags`),由 xgrammar 后端按 schema 约束生成。

- 用户按 strict-schema 风格定义 → xgrammar 生成精确约束;
- 否则约束可能不完整(但不会报错)。

`strict: False` 的工具即使进入 structural tag,`_get_function_parameters` 返回 `True`(任意 JSON),不附加 schema 约束。

---

## 七、各 parser 声明 `structural_tag_model`(0.24.0 新增,0.18.0 无)

| parser 文件 | `structural_tag_model` | 来源 |
| --- | --- | --- |
| `hermes_tool_parser.py:35` | `"hermes"` | vllm builtin |
| `minimax_m2_tool_parser.py:8` | `"minimax"` | vllm builtin |
| `llama_tool_parser.py:49` | `"llama"` | xgrammar |
| `kimi_k2_tool_parser.py:32` | `"kimi"` | xgrammar |
| `deepseekv3_tool_parser.py:31` | `"deepseek_r1"` | xgrammar |
| `deepseekv31_tool_parser.py:28` | `"deepseek_v3_1"` | xgrammar |
| `deepseekv32_tool_parser.py:56` | `"deepseek_v3_2"` | xgrammar |
| `deepseekv4_tool_parser.py:17` | `"deepseek_v4"` | xgrammar |
| `glm47_moe_tool_parser.py:11` | `"glm_4_7"` | xgrammar |
| `qwen3_engine_tool_parser.py:8` | `"qwen_3_coder"` | xgrammar |

测试佐证(`test_structural_tag_registry.py:207-209`):声明了 `structural_tag_model` 的 parser 均 `supports_required_and_named == False`。

---

## 八、总结:0.18.0 → 0.24.0 的本质变化

```
0.18.0:
  required/named → structured_outputs.json (schema 约束,早就有)
  auto          → 无约束,自由生成 + 文本提取
  无 strict 字段语义、无 structural tag、无全局开关

0.24.0:
  required/named → 若 parser 支持 structural tag,改走 structural_tag 约束
                   (supports_required_and_named=False,提取降级为 auto 式)
  auto          → strict:true 才生成 structural_tag;否则维持 0.18 行为
  新增:
    - VLLM_ENFORCE_STRICT_TOOL_CALLING 全局开关(默认 True)
    - structural_tag_registry.py 注册表(xgrammar 内置 + vllm hermes/minimax)
    - _apply_structural_tag 注入点(DelegatingParser)
    - ToolParser 新增 structural_tag_model / supports_required_and_named / __init_subclass__
    - Anthropic strict 字段贯通
    - serving 层实例化 parser 时传入 tools
```

**核心设计转变**:从"解析期修补"升级为"生成期约束(structural tag)"。`strict` 字段作为 `auto` 模式下的 opt-in 开关,`required`/named 则无条件启用,全局环境变量作为最终熔断开关。三者在 `structural_tag_registry.get_model_structural_tag` 中汇合判定。
