import json
from typing import Any

from pydantic import BaseModel, Field


class AgentBehavior(BaseModel):
    step_index: int | None = Field(
        default=None, ge=1, description="Vị trí behavior trong trajectory, bắt đầu từ 1."
    )
    agent_name: str | None = Field(
        default=None, description="Tên agent nếu dataset cung cấp."
    )
    role: str | None = Field(default=None, description="Vai trò hoặc speaker nếu có.")
    content: str | None = Field(
        default=None, description="Nội dung behavior khi không có input/output riêng."
    )
    input: dict[str, Any] | None = Field(
        default=None, description="Đầu vào model; chỉ TraceElephant có."
    )
    output: dict[str, Any] | None = Field(
        default=None, description="Đầu ra model; chỉ TraceElephant có."
    )
    tool_logs: list[dict[str, Any]] | None = Field(
        default=None, description="Log công cụ; chỉ TraceElephant có."
    )

    def format_content(self) -> str:
        """Format the behavior's main data and tool calls for slicing."""
        if self.input is not None or self.output is not None:
            parts = []
            if self.input is not None:
                parts.append(f"Input: {json.dumps(self.input, ensure_ascii=False)}")
            if self.output is not None:
                parts.append(f"Output: {json.dumps(self.output, ensure_ascii=False)}")
        else:
            parts = [self.content] if self.content else []

        if self.tool_logs:
            calls = []
            for log in self.tool_logs:
                function = log["function"]
                call = f"Sử dụng hàm {function['name']}"
                arguments = function.get("arguments")
                if arguments:
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    call += f" với tham số {arguments}"
                calls.append(call)
            parts.append("; ".join(calls))

        step = self.step_index if self.step_index is not None else "?"
        agent = self.agent_name or self.role or "Unknown"
        content = "\n".join(parts)
        return f"Bước {step}: Agent {agent}: {content}"


class Trajectory(BaseModel):
    dataset: str = Field(description="Tên dataset và biến thể nguồn.")
    task: str = Field(description="Đề bài; nguồn có thể là question hoặc instruction.")
    behaviors: list[AgentBehavior] = Field(
        description="Các behavior theo thứ tự trajectory."
    )
    ground_truth: str | None = Field(
        default=None, description="Đáp án chuẩn nếu dataset cung cấp."
    )
    mistake_agent: str | None = Field(
        default=None, description="Agent gây lỗi; AgentRX lấy từ failure gốc."
    )
    mistake_step: str | int | None = Field(
        default=None, description="Step gây lỗi theo quy ước 1-based."
    )
    mistake_reason: str | None = Field(
        default=None, description="Lý do lỗi; AgentRX lấy từ root cause."
    )
