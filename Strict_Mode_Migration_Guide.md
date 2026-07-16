# 0.18.0 → Strict Mode 迁移指南(1 对 1 修改)

> 目标:在 `vllm-releases-v0.18.0` 代码上,逐步修改使其实现 0.24.0 的 Strict Mode 等同功能。
>
> **适配说明**:0.18.0 的 serving 层直接调用 `tool_parser(tokenizer).adjust_request()`,而 0.24.0 改为先经 `DelegatingParser._apply_structural_tag` 再调 `ToolParser.adjust_request`。为最小化改动,本指南把 structural tag 注入逻辑**合并进 `ToolParser.adjust_request` 自身**,功能与 0.24.0 完全等价,仅代码组织位置不同(已在每处标注差异)。

所有路径相对于 `vllm-releases-v0.18.0/vllm-releases-v0.18.0/`。

---

## 修改清单总览

| # | 文件 | 操作 | 说明 |
|---|---|---|---|
| 1 | `vllm/envs.py` | 改 | 新增环境变量声明 + 解析 |
| 2 | `vllm/tool_parsers/structural_tag_registry.py` | **新建** | structural tag 注册表与判断逻辑 |
| 3 | `vllm/tool_parsers/abstract_tool_parser.py` | 改 | 核心改造:类属性、构造、get_structural_tag、adjust_request |
| 4 | `vllm/tool_parsers/utils.py` | 改 | 新增 `Tool` 类型别名(供 parser 引用) |
| 5 | `vllm/parser/abstract_parser.py` | 改 | `_parse_tool_calls` 接入 `supports_required_and_named` |
| 6 | `vllm/entrypoints/openai/engine/serving.py` | 改 | 实例化 parser 时传 `tools`;`_parse_tool_calls_from_content` 接入降级 |
| 7 | `vllm/entrypoints/serve/render/serving.py` | 改 | 实例化 parser 时传 `tools` |
| 8 | `vllm/entrypoints/anthropic/protocol.py` | 改 | `AnthropicTool` 新增 `strict` 字段 |
| 9 | `vllm/entrypoints/anthropic/serving.py` | 改 | `_convert_tools` 透传 `strict` |
| 10 | 各 tool parser(hermes 等) | 改 | 声明 `structural_tag_model` |

建议按 1→10 顺序执行,因为后续步骤依赖前面定义的符号。

---

## 步骤 1:`vllm/envs.py` — 新增全局开关

### 1a. 类属性声明(约第 181 行后插入)

**找到**:
```python
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
```

**改为**:
```python
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_ENFORCE_STRICT_TOOL_CALLING: bool = True
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
```

### 1b. 解析 lambda(约第 1340 行,`environment_variables` 字典内)

**找到**:
```python
    "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")
    ),
```

**在其后插入**:
```python
    "VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(
        os.getenv("VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")
    ),
    # Enforce function parameter schemas in structural-tag based tool calling.
    "VLLM_ENFORCE_STRICT_TOOL_CALLING": lambda: (
        os.getenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "True").lower()
        in ("true", "1")
    ),
```

---

## 步骤 2:新建 `vllm/tool_parsers/structural_tag_registry.py`

这是 0.24.0 的全新文件,直接从 0.24.0 拷贝即可。**前置依赖**:项目需安装 `xgrammar`(带 `StructuralTag`、`normalize_tool_choice`、`get_model_structural_tag`、`structural_tag` 子模块)。

若 xgrammar 版本不支持某些 builtin model key,可先只保留 hermes/minimax 两个 vllm 自有实现(删去 xgrammar 分支),功能对 hermes 类模型即完整。

文件内容(与 0.24.0 一致,关键结构):

```python
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

from openai.types.responses import FunctionTool
from openai.types.responses.response import ToolChoice as ResponsesToolChoice
from openai.types.responses.tool import Tool as ResponsesTool
from openai.types.responses.tool_choice_allowed import ToolChoiceAllowed
from openai.types.responses.tool_choice_function import ToolChoiceFunction
from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat, ConstStringFormat, JSONSchemaFormat, SequenceFormat,
    TagFormat, TagsWithSeparatorFormat, TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionToolsParam,
)

ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoiceParam
    | ResponsesToolChoice | None
)
SimplifiedToolChoice: TypeAlias = Literal["auto", "required", "forced"]
StructuralTagBuilder: TypeAlias = Callable[
    [list[FunctionToolParam], list[BuiltinToolParam], SimplifiedToolChoice, bool],
    StructuralTag,
]

XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({
    "llama", "kimi", "deepseek_r1", "deepseek_v3_1", "qwen_3_5",
    "qwen_3_coder", "qwen_3", "harmony", "deepseek_v3_2", "glm_4_7", "deepseek_v4",
})
VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"hermes"})
SUPPORTED_STRUCTURAL_TAG_MODELS = (
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
)

_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}


def register_vllm_structural_tag(model: str):
    def decorator(func):
        _VLLM_STRUCTURAL_TAG_REGISTRY[model] = func
        return func
    return decorator


def _any_tool_strict(tools):
    """任一 tool 的 strict == True 即返回 True(覆盖三种 API 的 tool 类型)"""
    for tool in tools:
        if isinstance(tool, FunctionTool) and tool.strict is True:
            return True
        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is True:
            return True
    return False


def get_model_structural_tag(model, tools, tool_choice, reasoning):
    """根据 model/tools/tool_choice 构建 StructuralTag,或返回 None"""
    if not tools or tool_choice == "none":
        return None
    # ★ auto 模式 opt-in:无 strict 则不约束
    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
    # required / named:跳过 strict 检查,总是生成 tag

    dumped_tools = [_dump_tool_for_xgrammar(t) for t in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

    if model in _VLLM_STRUCTURAL_TAG_REGISTRY:
        function_tools, builtin_tools, simplified = normalize_tool_choice(
            dumped_tools, dumped_tool_choice)
        return _VLLM_STRUCTURAL_TAG_REGISTRY[model](
            function_tools, builtin_tools, simplified, reasoning)

    if model not in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:
        raise ValueError(f"Unknown format type: {model}, supported: "
                         f"{sorted(SUPPORTED_STRUCTURAL_TAG_MODELS)}")
    return get_xgrammar_model_structural_tag(
        model=model, tools=dumped_tools,
        tool_choice=dumped_tool_choice, reasoning=reasoning)


def _dump_tool_for_xgrammar(tool):
    if isinstance(tool, FunctionTool):
        function = {"name": tool.name}
        if tool.description is not None:
            function["description"] = tool.description
        if tool.parameters is not None:
            function["parameters"] = tool.parameters
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}
    dumped = tool.model_dump(mode="json", exclude_none=True)
    return dumped


def _dump_tool_choice_for_xgrammar(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.model_dump(mode="json", exclude_none=True)
    if isinstance(tool_choice, ToolChoiceFunction):
        return {"type": "function", "function": {"name": tool_choice.name}}
    if isinstance(tool_choice, ToolChoiceAllowed):
        return {"type": "allowed_tools", "allowed_tools": {
            "mode": tool_choice.mode,
            "tools": [_dump_allowed_tool_ref_for_xgrammar(t) for t in tool_choice.tools]}}
    return tool_choice.model_dump(mode="json", exclude_none=True)


def _dump_allowed_tool_ref_for_xgrammar(tool_ref):
    if (tool_ref.get("type") == "function" and "function" not in tool_ref
            and "name" in tool_ref):
        return {"type": "function", "function": {"name": tool_ref["name"]}}
    return tool_ref


def _get_function_parameters(function):
    if getattr(function, "strict", None) is False:
        return True          # strict=False → 放松,无 schema 约束
    return function.parameters if function.parameters is not None else True


# ── vllm 自有 builder:hermes ──────────────────────────────────────
def _hermes_tool_tags(tools):
    arguments_field_prefix = '", "arguments": '
    formats = [
        ('<tool_call>\n{"name": "', "}\n</tool_call>"),
        ('<tool_call>{"name": "', "}</tool_call>"),
    ]
    return [
        TagFormat(
            begin=begin + tool.function.name + arguments_field_prefix,
            content=JSONSchemaFormat(json_schema=_get_function_parameters(tool.function)),
            end=end,
        )
        for tool in tools for begin, end in formats
    ]


@register_vllm_structural_tag("hermes")
def get_hermes_structural_tag(tools, builtin_tools, tool_choice, reasoning):
    del builtin_tools, reasoning
    trigger = "<tool_call>"
    if tool_choice == "auto":
        tags = _hermes_tool_tags(tools)
        suffix = (TriggeredTagsFormat(triggers=[trigger], tags=tags)
                  if tags else AnyTextFormat())
    elif tool_choice == "forced":
        suffix = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools), separator="",
            at_least_one=True, stop_after_first=True)
    else:
        suffix = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools), separator="", at_least_one=True)
    return StructuralTag(format=suffix)
```

> **说明**:minimax builder 结构类似,如需支持 minimax 可从 0.24.0 `structural_tag_registry.py:272-347` 完整拷贝 `_minimax_tool_tags` + `get_minimax_structural_tag`。本指南以 hermes 为代表。

---

## 步骤 3:`vllm/tool_parsers/abstract_tool_parser.py` — 核心改造

### 3a. 顶部新增 import

**找到**(文件头 import 区):
```python
import importlib
import os
from collections.abc import Callable, Sequence
from functools import cached_property

from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.logger import init_logger
from vllm.sampling_params import (
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import get_json_schema_from_tools
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path
```

**改为**(对齐 0.24.0:统一从 `openai.types.responses` 导入、加 `envs`、加 `Tool`):
```python
import importlib
import json
import os
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfig,
    ResponseTextConfig,
)
from openai.types.responses.function_tool import FunctionTool

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.sampling_params import (
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import Tool, get_json_schema_from_tools
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

__all__ = ["Tool"]
```

> 注:`ResponseTextConfig` 的导入路径从 `vllm.entrypoints.openai.responses.protocol` 改为 `openai.types.responses`(0.24.0 的写法)。若 0.18.0 的 openai 版本不支持,保留原路径即可。

### 3b. 类属性 + `__init_subclass__` + 构造函数

**找到**:
```python
class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def __init__(self, tokenizer: TokenizerLike):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer
```

**改为**:
```python
class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    # When True (default), the serving layer uses the standard JSON-based
    # parsing for tool_choice="required" and named function tool_choice.
    supports_required_and_named: bool = True
    # xgrammar builtin structural tag model key. Subclasses set this when
    # their parsed tool-call syntax matches a builtin xgrammar format.
    structural_tag_model: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (cls.structural_tag_model is not None
                and envs.VLLM_ENFORCE_STRICT_TOOL_CALLING):
            cls.supports_required_and_named = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer
        if tools:
            self.tools: list[ChatCompletionToolsParam | FunctionTool] = [
                tool
                for tool in tools
                if isinstance(tool, (ChatCompletionToolsParam, FunctionTool))
            ]
        else:
            self.tools = []
```

### 3c. `adjust_request` 改造(合并 structural tag 注入 + JSON schema 短路)

> **与 0.24.0 的差异**:0.24.0 把 structural tag 注入放在 `DelegatingParser._apply_structural_tag`,这里因为 0.18.0 serving 层直接调 `ToolParser.adjust_request`,所以**合并到一处**,功能等价。

**找到**:
```python
    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Static method that used to adjust the request parameters.
        """
        if not request.tools:
            return request
        json_schema_from_tool = get_json_schema_from_tools(
            tool_choice=request.tool_choice, tools=request.tools
        )
        # Set structured output params for tool calling
        if json_schema_from_tool is not None:
            if isinstance(request, ChatCompletionRequest):
                # tool_choice: "Forced Function" or "required" will override
                # structured output json settings to make tool calling work correctly
                request.structured_outputs = StructuredOutputsParams(
                    json=json_schema_from_tool  # type: ignore[call-arg]
                )
                request.response_format = None
            if isinstance(request, ResponsesRequest):
                request.text = ResponseTextConfig()
                request.text.format = ResponseFormatTextJSONSchemaConfig(
                    name="tool_calling_response",
                    schema=json_schema_from_tool,
                    type="json_schema",
                    description="Response format for tool calling",
                    strict=True,
                )

        return request
```

**改为**:
```python
    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        if not request.tools:
            return request

        # ── 1) 优先尝试 structural tag(等价于 0.24.0 DelegatingParser._apply_structural_tag)
        if self.structural_tag_model is not None and envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            need_tool_calling = (
                request.tool_choice == "auto"
                or request.tool_choice == "required"
                or isinstance(
                    request.tool_choice,
                    (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
                )
            )
            if need_tool_calling:
                structure_tag = self.get_structural_tag(request, reasoning=False)
                if structure_tag is not None:
                    structural_tag = json.dumps(structure_tag.model_dump())
                    request.structured_outputs = StructuredOutputsParams(
                        structural_tag=structural_tag,
                    )
                    if isinstance(request, ResponsesRequest):
                        request.text = None
                    else:
                        request.response_format = None
                    return request   # 已用 structural tag,跳过 JSON-schema 路径

        # ── 2) 短路:若 structured_outputs 已被上层(或上一步)设了 structural_tag,直接返回
        structured_outputs = getattr(request, "structured_outputs", None)
        if (structured_outputs is not None
                and structured_outputs.structural_tag is not None):
            return request

        # ── 3) 原有 JSON-schema 路径(named/required)
        json_schema_from_tool = get_json_schema_from_tools(
            tool_choice=request.tool_choice, tools=request.tools
        )
        if json_schema_from_tool is not None:
            if isinstance(request, ChatCompletionRequest):
                request.structured_outputs = StructuredOutputsParams(
                    json=json_schema_from_tool  # type: ignore[call-arg]
                )
                request.response_format = None
            if isinstance(request, ResponsesRequest):
                request.text = ResponseTextConfig(
                    format=ResponseFormatTextJSONSchemaConfig(
                        type="json_schema",
                        name="tool_calling_response",
                        schema=json_schema_from_tool,
                        strict=True,
                    )
                )

        return request
```

> 需要 import `ChatCompletionNamedToolChoiceParam` 与 `ToolChoiceFunction`。在文件头补充:
> ```python
> from openai.types.responses.tool_choice_function import ToolChoiceFunction
> from vllm.entrypoints.openai.chat_completion.protocol import (
>     ChatCompletionNamedToolChoiceParam, ChatCompletionRequest, ChatCompletionToolsParam,
> )
> ```

### 3d. 新增 `get_structural_tag` 方法

在 `adjust_request` 之后、`extract_tool_calls` 之前插入:

```python
    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
    ):
        if self.structural_tag_model is None:
            return None
        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            return None
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        return get_model_structural_tag(
            model=self.structural_tag_model,
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=reasoning,
        )
```

### 3e. 所有子类的 `__init__` 签名对齐

0.18.0 各 parser 的 `__init__(self, tokenizer)` 需改为 `__init__(self, tokenizer, tools=None)`,并调 `super().__init__(tokenizer, tools)`。

以 `hermes_tool_parser.py` 为例(见步骤 10),其余 parser 同理。

---

## 步骤 4:`vllm/tool_parsers/utils.py` — 新增 `Tool` 类型别名

**找到**(约第 14 行):
```python
from openai.types.responses.tool import Tool
```

**在其后新增**:
```python
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
```
(若已存在则跳过)

然后在 import 块之后、`logger = init_logger(__name__)` 之前新增:
```python
Tool: TypeAlias = ChatCompletionToolsParam | ResponsesTool
```

并确保文件头有 `from typing import TypeAlias`(0.18.0 用 `from typing import Any`,补上 `TypeAlias`)。

> 这样 `abstract_tool_parser.py` 的 `from vllm.tool_parsers.utils import Tool` 才能生效。

---

## 步骤 5:`vllm/parser/abstract_parser.py` — 接入 `supports_required_and_named`

### 5a. `_parse_tool_calls`(Responses 路径,约第 388 行)

**找到**:
```python
    def _parse_tool_calls(
        self,
        request: ResponsesRequest,
        content: str | None,
        enable_auto_tools: bool,
    ) -> tuple[list[FunctionCall], str | None]:
        function_calls: list[FunctionCall] = []

        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            # Forced Function Call (Responses API style)
            assert content is not None
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            return function_calls, None  # Clear content since tool is called.

        if request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            # Forced Function Call (Chat Completion API style)
            assert content is not None
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            return function_calls, None  # Clear content since tool is called.

        if request.tool_choice == "required":
            # Required tool calls - parse JSON
            assert content is not None
            tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
            function_calls.extend(
                FunctionCall(
                    name=tool_call.name,
                    arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                )
                for tool_call in tool_calls
            )
            return function_calls, None  # Clear content since tool is called.

        if (
            self._tool_parser is not None
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            # Automatic Tool Call Parsing
            ...
```

**改为**(加入 `supports_required_and_named` 降级判断):
```python
    def _parse_tool_calls(
        self,
        request: ResponsesRequest,
        content: str | None,
        enable_auto_tools: bool,
    ) -> tuple[list[FunctionCall], str | None]:
        function_calls: list[FunctionCall] = []

        supports = (self._tool_parser.supports_required_and_named
                    if self._tool_parser else True)
        is_named = request.tool_choice and isinstance(
            request.tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam))
        is_required = request.tool_choice == "required"
        is_auto = enable_auto_tools and (
            request.tool_choice == "auto"
            or request.tool_choice is None
            or (not supports and (is_named or is_required))   # ★ 降级
        )

        if is_named and supports:
            assert content is not None
            name = (request.tool_choice.name
                    if isinstance(request.tool_choice, ToolChoiceFunction)
                    else request.tool_choice.function.name)
            function_calls.append(FunctionCall(name=name, arguments=content))
            return function_calls, None

        if is_required and supports:
            assert content is not None
            tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
            function_calls.extend(
                FunctionCall(name=tc.name,
                             arguments=json.dumps(tc.parameters, ensure_ascii=False))
                for tc in tool_calls)
            return function_calls, None

        if is_auto and self._tool_parser is not None:
            # Automatic Tool Call Parsing(含 structural-tag parser 的降级路径)
            tool_call_info = self._tool_parser.extract_tool_calls(
                content if content is not None else "", request=request,
            )
            # ...保留原有提取逻辑不变...
```

> 流式路径(`_extract_tool_calls_streaming` 在 0.18.0 中可能位于 serving 层)同理:对 `supports_required_and_named=False` 的 parser,named/required 走 `extract_tool_calls_streaming`。

---

## 步骤 6:`vllm/entrypoints/openai/engine/serving.py` — 传 tools + 降级

### 6a. adjust_request 调用点(约第 925 行)

**找到**:
```python
                # TODO: Update adjust_request to accept ResponsesRequest
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]
```

**改为**:
```python
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer, request.tools).adjust_request(request=request)  # type: ignore[arg-type]
```

### 6b. `_parse_tool_calls_from_content`(约第 1104 行)

**找到** `tool_parser_cls` 的类型注解与 auto 分支:
```python
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
        ...
        elif (
            tool_parser_cls
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            ...
            tool_parser = tool_parser_cls(tokenizer)
```

**改为**:
```python
        tool_parser_cls: Callable[[TokenizerLike, list], ToolParser] | None,
        ...
        # 判断是否需要降级
        supports = True  # 默认
        is_named = request.tool_choice and isinstance(
            request.tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam))
        is_required = request.tool_choice == "required"
        # named/required 仍走标准提取(除非 parser 不支持)
        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            ...原有 named 逻辑...
        elif request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam):
            ...原有 named 逻辑...
        elif request.tool_choice == "required":
            ...原有 required 逻辑...
        elif (
            tool_parser_cls
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            ...
            tool_parser = tool_parser_cls(tokenizer, request.tools)
```

> 核心改动:`tool_parser_cls(tokenizer)` → `tool_parser_cls(tokenizer, request.tools)`。
>
> 若要让 structural-tag parser 的 required/named 也降级到 auto 提取,需在此处也加 `supports_required_and_named` 判断(参考步骤 5 的模式)。最小化改动下,可先只加 tools 传参,降级逻辑后续补。

---

## 步骤 7:`vllm/entrypoints/serve/render/serving.py` — 传 tools

**找到**(约第 553 行):
```python
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]
```

**改为**:
```python
                request = tool_parser(tokenizer, request.tools).adjust_request(request=request)  # type: ignore[arg-type]
```

---

## 步骤 8:`vllm/entrypoints/anthropic/protocol.py` — 新增 strict 字段

**找到**:
```python
class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        ...
```

**改为**:
```python
class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    strict: bool | None = None
    defer_loading: bool | None = None

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        ...
```

---

## 步骤 9:`vllm/entrypoints/anthropic/serving.py` — 透传 strict

**找到 `_convert_tools` 方法**(搜 `def _convert_tools`):
```python
    def _convert_tools(
        cls,
        anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
        req: ChatCompletionRequest,
    ) -> None:
        """Convert Anthropic tools to OpenAI format"""
        if anthropic_request.tools is None:
            return

        tools = []
        for tool in anthropic_request.tools:
            tools.append(
                ChatCompletionToolsParam.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )
            )
        ...
```

**改为**(加 `strict` 与 `defer_loading`):
```python
        tools = []
        for tool in anthropic_request.tools:
            tools.append(
                ChatCompletionToolsParam.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                            "strict": tool.strict,
                            "defer_loading": tool.defer_loading,
                        },
                    }
                )
            )
```

> 确保 `ChatCompletionToolsParam` 的 `function` 子模型接受 `strict` 字段。0.18.0 的 `ChatCompletionToolsParam` 若基于 openai SDK,通常已有 `strict`;若为 vllm 自定义,需在 protocol 中补字段。

---

## 步骤 10:各 tool parser 声明 `structural_tag_model` + 对齐 `__init__`

### 10a. `vllm/tool_parsers/hermes_tool_parser.py`

**找到**:
```python
class Hermes2ProToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
```

**改为**:
```python
class Hermes2ProToolParser(ToolParser):
    structural_tag_model = "hermes"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
```

> 需 import `Tool`:`from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser`(0.24.0 把 `Tool` re-export 到 abstract_tool_parser 的 `__all__`)。

### 10b. 其他 parser(按需)

对每个要启用 strict 的 parser,加一行类属性 + 改 `__init__` 签名:

| 文件 | 添加 |
|---|---|
| `llama_tool_parser.py` | `structural_tag_model = "llama"` |
| `kimi_k2_tool_parser.py` | `structural_tag_model = "kimi"` |
| `deepseekv3_tool_parser.py` | `structural_tag_model = "deepseek_r1"` |
| `deepseekv31_tool_parser.py` | `structural_tag_model = "deepseek_v3_1"` |
| `deepseekv32_tool_parser.py` | `structural_tag_model = "deepseek_v3_2"` |
| `glm47_moe_tool_parser.py` | `structural_tag_model = "glm_4_7"` |

每个的 `__init__(self, tokenizer)` → `__init__(self, tokenizer, tools=None)`,`super().__init__(tokenizer)` → `super().__init__(tokenizer, tools)`。

> **前提**:对应的 model key 必须在 `structural_tag_registry.py` 的注册表中(xgrammar 内置或 vllm 注册)。若 xgrammar 版本不支持某 key,该 parser 不要声明,否则运行时报 `Unknown format type`。

---

## 验证检查清单

完成上述修改后,按以下顺序验证:

1. **环境变量**:
   ```python
   import vllm.envs as envs
   assert envs.VLLM_ENFORCE_STRICT_TOOL_CALLING is True  # 默认
   ```

2. **类属性自动联动**:
   ```python
   from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
   assert Hermes2ProToolParser.structural_tag_model == "hermes"
   assert Hermes2ProToolParser.supports_required_and_named is False  # __init_subclass__ 触发
   ```

3. **auto + 无 strict → 无约束**:
   ```python
   from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
   tag = get_model_structural_tag("hermes", [tool_without_strict], "auto", False)
   assert tag is None
   ```

4. **auto + strict:true → 有 tag**:
   ```python
   tag = get_model_structural_tag("hermes", [tool_with_strict_true], "auto", False)
   assert tag is not None
   ```

5. **required → 始终有 tag**(无需 strict):
   ```python
   tag = get_model_structural_tag("hermes", [tool_without_strict], "required", False)
   assert tag is not None
   ```

6. **全局开关关闭**:
   ```bash
   VLLM_ENFORCE_STRICT_TOOL_CALLING=false python -c "..."
   # adjust_request 不应设置 structural_tag,退回 JSON-schema 路径
   ```

7. **端到端**:用 hermes 模型发起 `tool_choice="auto"` + `strict: true` 的请求,确认生成期输出被约束为合法 `<tool_call>` JSON。

---

## 与 0.24.0 的残留差异(可接受)

| 方面 | 本指南(0.18.0 改造) | 0.24.0 原版 |
|---|---|---|
| structural tag 注入位置 | 合并进 `ToolParser.adjust_request` | 拆分到 `DelegatingParser._apply_structural_tag` |
| serving 层实例化 | `tool_parser(tokenizer, tools)` | `parser(tokenizer, request.tools, model_config=...)` 经 DelegatingParser |
| 流式降级 | 需手动在 serving 流式路径补 `supports_required_and_named` | 已在 `DelegatingParser._extract_tool_calls_streaming` 统一 |
| minimax builder | 可选拷贝 | 内置 |
| `Tool` re-export | 经 `utils.py` | 同时在 `abstract_tool_parser.__all__` |

功能层面:strict 字段语义、auto opt-in、required/named 无条件约束、全局开关、三 API 贯通——**全部等价**。
