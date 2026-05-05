from __future__ import annotations

from ._shared import *

class Parser:
    def _empty_parse_result(self) -> dict:
        return {
            "is_function": False,
            "function_name": None,
            "component_or_param": None,
            "multiple_sources": None,
            "single_source": None,
            "selection": None,
        }

    def parse_input_string(self, spec: str) -> dict:
        spec = spec.strip()
        if not spec:
            return self._empty_parse_result()

        if ":" in spec and (spec.startswith("fn:") or ".py:" in spec):
            return self._parse_as_function_reference(spec)

        return self._parse_as_non_function(spec)


    def _parse_as_function_reference(self, full_ref: str) -> dict:
        return {
            "is_function": True,
            "function_name": full_ref.strip(),
            "component_or_param": None,
            "multiple_sources": None,
            "single_source": None,
            "selection": None,
        }


    def _parse_as_non_function(self, spec: str) -> dict:
        if spec.startswith(":"):
            remainder = spec[1:].strip()
            return self._parse_input_sources(remainder)

        if ":" in spec:
            parts = spec.split(":", 1)
            component_name = parts[0].strip()
            remainder = parts[1].strip()

            selection = self._parse_selection_expression(remainder)

            if selection is not None and selection["selector"]["type"] == "include":
                multiple_sources = copy.deepcopy(selection["selector"]["sources"])
                return {
                    "is_function": False,
                    "function_name": None,
                    "component_or_param": component_name,
                    "multiple_sources": multiple_sources,
                    "single_source": None,
                    "selection": selection,
                }

            if selection is not None:
                return {
                    "is_function": False,
                    "function_name": None,
                    "component_or_param": component_name,
                    "multiple_sources": None,
                    "single_source": None,
                    "selection": selection,
                }

            single_source = self._parse_one_custom_item(remainder)
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": component_name,
                "multiple_sources": None,
                "single_source": single_source,
                "selection": None,
            }

        return {
            "is_function": False,
            "function_name": None,
            "component_or_param": spec,
            "multiple_sources": None,
            "single_source": None,
            "selection": None,
        }


    def _parse_input_sources(self, remainder: str) -> dict:
        remainder = remainder.strip()
        if not remainder:
            return self._empty_parse_result()

        selection = self._parse_selection_expression(remainder)

        if selection is not None and selection["selector"]["type"] == "include":
            multiple_sources = copy.deepcopy(selection["selector"]["sources"])
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": multiple_sources,
                "single_source": None,
                "selection": selection,
            }

        if selection is not None:
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": None,
                "single_source": None,
                "selection": selection,
            }

        single_source = self._parse_one_custom_item(remainder)
        return {
            "is_function": False,
            "function_name": None,
            "component_or_param": None,
            "multiple_sources": None,
            "single_source": single_source,
            "selection": None,
        }

    def _parse_selection_expression(self, expr: str) -> Optional[dict]:
        expr = expr.strip()
        if not expr:
            return None

        if expr.startswith("*"):
            selector, remainder = self._parse_star_selector(expr)
            global_index = None
            if remainder:
                if not remainder.startswith("?"):
                    raise ValueError(f"Unexpected input selector suffix: {remainder}")
                global_index = self._parse_global_index(remainder[1:])
            return {
                "mode": "timeline",
                "selector": selector,
                "global_index": global_index,
            }

        if expr.startswith("("):
            close_idx = self._find_matching(expr, 0, "(", ")")
            if close_idx is None:
                return None
            inside = expr[1:close_idx].strip()
            remainder = expr[close_idx + 1:].strip()
            if remainder and not remainder.startswith("?"):
                return None
            sources = self._parse_multiple_sources(inside)
            global_index = self._parse_global_index(remainder[1:]) if remainder else None
            return {
                "mode": "timeline",
                "selector": {
                    "type": "include",
                    "sources": sources,
                },
                "global_index": global_index,
            }

        return None

    def _parse_star_selector(self, expr: str) -> tuple:
        remainder = expr[1:].strip()
        if not remainder:
            return {"type": "all"}, ""

        if remainder.startswith("!"):
            rest = remainder[1:].strip()
            if rest.startswith("("):
                close_idx = self._find_matching(rest, 0, "(", ")")
                if close_idx is None:
                    raise ValueError(f"Invalid negative selector: {expr}")
                inside = rest[1:close_idx].strip()
                excluded = [part.strip() for part in self._split_by_top_level_comma(inside) if part.strip()]
                return {"type": "exclude", "components": excluded}, rest[close_idx + 1:].strip()

            if not rest:
                raise ValueError(f"Invalid negative selector: {expr}")
            if "?" in rest:
                name, suffix = rest.split("?", 1)
                return {"type": "exclude", "components": [name.strip()]}, "?" + suffix
            return {"type": "exclude", "components": [rest.strip()]}, ""

        if remainder.startswith("?"):
            return {"type": "all"}, remainder

        raise ValueError(f"Invalid '*' selector syntax: {expr}")

    def _find_matching(self, text: str, start_idx: int, opener: str, closer: str) -> Optional[int]:
        depth = 0
        bracket_depth = 0
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
            elif bracket_depth == 0 and ch == opener:
                depth += 1
            elif bracket_depth == 0 and ch == closer:
                depth -= 1
                if depth == 0:
                    return idx
        return None

    def _parse_global_index(self, spec: str):
        spec = spec.strip()
        if not spec or spec == "~":
            return "~"

        parts = spec.split("~")
        if len(parts) == 1:
            return self._parse_global_endpoint(parts[0])
        if len(parts) == 2:
            return (
                self._parse_global_endpoint(parts[0]) if parts[0].strip() else None,
                self._parse_global_endpoint(parts[1]) if parts[1].strip() else None,
            )
        raise ValueError(f"Invalid global range: {spec}")

    def _parse_global_endpoint(self, token: str):
        token = token.strip()
        if not token:
            return None
        try:
            return int(token)
        except ValueError:
            pass

        item = self._parse_one_custom_item(token)
        if not item.get("component"):
            raise ValueError(f"Invalid global range endpoint: {token}")
        if item.get("fields"):
            raise ValueError(f"Global range endpoints cannot include fields: {token}")
        idx = item.get("index")
        if idx in (None, ""):
            idx = -1
        if not isinstance(idx, int):
            raise ValueError(f"Global range endpoint must identify one message: {token}")
        return {
            "type": "anchor",
            "component": item["component"],
            "index": idx,
        }


    def _parse_multiple_sources(self, content: str) -> list:
        segments = [seg.strip() for seg in self._split_by_top_level_comma(content)]
        results = []
        for seg in segments:
            results.append(self._parse_one_custom_item(seg))
        return results


    def _split_by_top_level_comma(self, content: str) -> list:
        parts = []
        current = []
        nesting = 0
        for ch in content:
            if ch in ['(', '[']:
                nesting += 1
                current.append(ch)
            elif ch in [')', ']']:
                nesting -= 1
                current.append(ch)
            elif ch == ',' and nesting == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts


    def _parse_one_custom_item(self, item_str: str) -> dict:
        item_str = item_str.strip()
        result = {
            "component": None,
            "index": None,
            "fields": None,
        }

        fields_part = None
        bracket_start = item_str.find("[")
        bracket_end = item_str.find("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
            fields_part = item_str[bracket_start + 1 : bracket_end].strip()
            item_str = item_str[:bracket_start].strip() + item_str[bracket_end+1:].strip()

            if fields_part:
                field_list = [f.strip() for f in fields_part.split(",") if f.strip()]
                if field_list:
                    result["fields"] = field_list

        if "?" in item_str:
            parts = item_str.split("?")
            maybe_comp = parts[0].strip()
            if maybe_comp:
                result["component"] = maybe_comp
            index_part = parts[1].strip()

            if index_part == "~":
                result["index"] = "~"
            elif "~" in index_part:
                range_parts = index_part.split("~")
                if len(range_parts) == 2:
                    start_str = range_parts[0].strip()
                    end_str = range_parts[1].strip()
                    start = int(start_str) if start_str else None
                    end = int(end_str) if end_str else None
                    result["index"] = (start, end)
                elif len(range_parts) == 1:
                    range_str = range_parts[0].strip()
                    try:
                        index_val = int(range_str)
                        if index_part.startswith("~"):
                            result["index"] = (None, index_val)
                        elif index_part.endswith("~"):
                            result["index"] = (index_val, None)
                    except ValueError:
                        pass
            elif index_part:
                try:
                    idx_val = int(index_part)
                    result["index"] = idx_val
                except ValueError:
                    pass
        elif item_str:
            result["component"] = item_str

        return result




