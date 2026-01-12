from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Constant matching the schema version requirement
SCHEMA_VERSION = "1.0"


class PreprocessorType(str, Enum):
    INCLUDE = "include"
    SHELL = "shell"
    WEB = "web"
    VARIABLE = "variable"


class IncludeSyntax(str, Enum):
    DIRECTIVE = "directive"
    BACKTICKS = "backticks"


class PreprocessorItem(BaseModel):
    """
    Represents a single item processed by the preprocessor (e.g., a file inclusion,
    shell command execution, etc.).
    """
    model_config = ConfigDict(extra="forbid")

    type: PreprocessorType
    chars: int = Field(ge=0, description="Number of characters contributed by this item")
    line_in_prompt: Optional[int] = Field(None, ge=1, description="Line number in the source prompt file")
    
    # Fields specific to 'include' type
    syntax: Optional[IncludeSyntax] = None
    source: Optional[str] = None
    include_many: Optional[bool] = None
    
    # Fields specific to 'shell' type
    command: Optional[str] = None
    
    # Fields specific to 'web' type
    url: Optional[str] = None
    
    # Fields specific to 'variable' type
    name: Optional[str] = None


class PreprocessorSummary(BaseModel):
    """Summary statistics for preprocessor operations."""
    model_config = ConfigDict(extra="forbid")

    include_count: int = Field(0, ge=0)
    include_chars: int = Field(0, ge=0)
    shell_count: int = Field(0, ge=0)
    shell_chars: int = Field(0, ge=0)
    web_count: int = Field(0, ge=0)
    web_chars: int = Field(0, ge=0)
    variable_count: int = Field(0, ge=0)
    variable_chars: int = Field(0, ge=0)


class PreprocessorSummaryExtra(BaseModel):
    """Additional summary statistics for advanced preprocessor features."""
    model_config = ConfigDict(extra="forbid")

    include_many_count: int = Field(0, ge=0)
    include_many_chars: int = Field(0, ge=0)


class FewShotExample(BaseModel):
    """Represents a single few-shot example used in the prompt."""
    model_config = ConfigDict(extra="forbid")

    example_id: str
    chars: int = Field(ge=0)
    pinned: Optional[bool] = False
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class PromptBreakdown(BaseModel):
    """Detailed breakdown of character counts within the prompt construction."""
    model_config = ConfigDict(extra="forbid")

    pdd_system_prompt_chars: int = Field(0, ge=0)
    devunit_prompt_chars: int = Field(0, ge=0)
    prepended_chars: int = Field(0, ge=0)
    appended_chars: int = Field(0, ge=0)
    
    preprocessor_total_chars: int = Field(0, ge=0)
    preprocessor_items: List[PreprocessorItem] = Field(default_factory=list)
    preprocessor_summary: PreprocessorSummary
    preprocessor_summary_extra: Optional[PreprocessorSummaryExtra] = None
    
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)
    few_shot_total_chars: int = Field(0, ge=0)


class ApiStructure(BaseModel):
    """Breakdown of characters as sent to the LLM API structure."""
    model_config = ConfigDict(extra="forbid")

    system_prompt_chars: int = Field(0, ge=0)
    user_message_chars: int = Field(0, ge=0)
    assistant_prefill_chars: int = Field(0, ge=0)
    other_chars: int = Field(0, ge=0)


class Input(BaseModel):
    """Information about the input sent to the model."""
    model_config = ConfigDict(extra="forbid")

    total_chars: int = Field(ge=0)
    api_structure: Optional[ApiStructure] = None
    prompt_breakdown: Optional[PromptBreakdown] = None


class Output(BaseModel):
    """Information about the output received from the model."""
    model_config = ConfigDict(extra="forbid")

    response_chars: int = Field(ge=0)
    response_tokens_reported: Optional[int] = Field(None, ge=0)
    response_tokens_estimated: Optional[int] = Field(None, ge=0)
    prompt_tokens_reported: Optional[int] = Field(None, ge=0)


class Provenance(BaseModel):
    """Metadata about how and when this context map was generated."""
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    model: str
    provider: str
    prompt_file: str
    duration_ms: Optional[int] = Field(None, ge=0)
    pdd_version: Optional[str] = None

    @field_validator('timestamp_utc')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            # Basic ISO 8601 check
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("timestamp_utc must be a valid ISO 8601 string")
        return v


class ContextMap(BaseModel):
    """
    Root model for the PDD Context Map.
    Serves as the single source of truth for context map data structures.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    generation_id: str = Field(description="UUID format identifier")
    provenance: Provenance
    input: Input
    output: Output

    @field_validator('generation_id')
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("generation_id must be a valid UUID string")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, excluding None values."""
        return self.model_dump(exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string, excluding None values."""
        return self.model_dump_json(indent=indent, exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextMap:
        """Parse from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> ContextMap:
        """Parse from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, path: str | Path) -> ContextMap:
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        """Write to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def generate_sample(cls) -> ContextMap:
        """Create a realistic sample ContextMap with all fields populated."""
        
        # 1. Create Preprocessor Items
        items = [
            PreprocessorItem(
                type=PreprocessorType.INCLUDE,
                chars=1500,
                line_in_prompt=10,
                syntax=IncludeSyntax.DIRECTIVE,
                source="src/main.py"
            ),
            PreprocessorItem(
                type=PreprocessorType.INCLUDE,
                chars=800,
                line_in_prompt=12,
                syntax=IncludeSyntax.BACKTICKS,
                source="src/utils.py",
                include_many=True
            ),
            PreprocessorItem(
                type=PreprocessorType.SHELL,
                chars=200,
                line_in_prompt=25,
                command="ls -la"
            ),
            PreprocessorItem(
                type=PreprocessorType.WEB,
                chars=5000,
                line_in_prompt=30,
                url="https://docs.python.org/3/library/json.html"
            ),
            PreprocessorItem(
                type=PreprocessorType.VARIABLE,
                chars=50,
                line_in_prompt=5,
                name="USER_NAME"
            )
        ]

        # 2. Create Summaries
        summary = PreprocessorSummary(
            include_count=2, include_chars=2300,
            shell_count=1, shell_chars=200,
            web_count=1, web_chars=5000,
            variable_count=1, variable_chars=50
        )
        
        summary_extra = PreprocessorSummaryExtra(
            include_many_count=1,
            include_many_chars=800
        )

        # 3. Create Few Shot Examples
        few_shots = [
            FewShotExample(example_id="ex_001", chars=400, pinned=True, quality_score=0.95),
            FewShotExample(example_id="ex_002", chars=350, pinned=False, quality_score=0.88)
        ]

        # 4. Create Prompt Breakdown
        breakdown = PromptBreakdown(
            pdd_system_prompt_chars=500,
            devunit_prompt_chars=200,
            prepended_chars=100,
            appended_chars=50,
            preprocessor_total_chars=7550,  # Sum of items
            preprocessor_items=items,
            preprocessor_summary=summary,
            preprocessor_summary_extra=summary_extra,
            few_shot_examples=few_shots,
            few_shot_total_chars=750
        )

        # 5. Create API Structure
        api_struct = ApiStructure(
            system_prompt_chars=600,
            user_message_chars=8400,
            assistant_prefill_chars=0,
            other_chars=150
        )

        # 6. Create Input/Output/Provenance
        input_obj = Input(
            total_chars=9150,
            api_structure=api_struct,
            prompt_breakdown=breakdown
        )

        output_obj = Output(
            response_chars=1200,
            response_tokens_reported=300,
            response_tokens_estimated=310,
            prompt_tokens_reported=2500
        )

        provenance = Provenance(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model="gpt-4-turbo",
            provider="openai",
            prompt_file="prompts/code_review.pdd",
            duration_ms=4500,
            pdd_version="2.1.0"
        )

        return cls(
            schema_version=SCHEMA_VERSION,
            generation_id=str(uuid.uuid4()),
            provenance=provenance,
            input=input_obj,
            output=output_obj
        )