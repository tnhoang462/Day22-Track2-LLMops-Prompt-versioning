"""Step 4 - Custom Guardrails AI validators (PII redaction + JSON repair)."""

from __future__ import annotations

import json
import re

from guardrails import Guard, OnFailAction
from guardrails.validators import (
    FailResult,
    PassResult,
    Validator,
    register_validator,
)


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detect emails, phone numbers, SSNs, and credit-card numbers; redact in place."""

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value, metadata):
        text = value if isinstance(value, str) else str(value)
        redacted = text
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.findall(pattern, text):
                redacted = redacted.replace(match, f"[{pii_type}_REDACTED]")
                found.append((pii_type, match))

        if found:
            print(f"  Redacted {len(found)} PII items: {[p[0] for p in found]}")
            return FailResult(
                error_message=f"PII detected: {[p[0] for p in found]}",
                fix_value=redacted,
            )
        return PassResult()


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validate JSON; auto-repair markdown fences, single quotes, trailing commas."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value, metadata):
        text = value if isinstance(value, str) else str(value)
        try:
            parsed = json.loads(text)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(text)
            parsed = json.loads(repaired_text)
            print(f"  JSON repaired successfully")
            return FailResult(
                error_message="Original JSON was malformed but auto-repaired",
                fix_value=json.dumps(parsed, indent=2),
            )
        except json.JSONDecodeError as e:
            fallback = json.dumps({"error": f"invalid JSON: {e}", "raw": text[:200]})
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=fallback,
            )


def demo_pii_guard() -> None:
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")
        print(f"  Passed: {result.validation_passed}")


def demo_json_guard() -> None:
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",      '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",   "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",  '{"key": "value",}'),
        ("Truly invalid",   "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "Pass" if result.validation_passed else "Fixed/Failed"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:200]}")


def main() -> None:
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
