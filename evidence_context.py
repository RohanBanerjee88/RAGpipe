"""Reconstruct bounded, source-local procedure context after passage retrieval."""

import re


MAX_CONTEXT_TOKENS = 1536


def procedure_query(query):
    return bool(re.search(r"\b(how|steps?|procedure|commands?|scripts?|code|example|"
                          r"configure|install|run|fit|save|read)\b", query, re.I))


def context_blocks(sections, raw_text=None):
    from markdown_it import MarkdownIt
    headings = []
    if raw_text is not None:
        headings = [token.map[0] + 1 for token in MarkdownIt().parse(raw_text)
                    if token.type == "heading_open" and token.map]
    result = []
    group = -1
    previous_title = None
    for title, location, text, kind in sections:
        if title != previous_title:
            group += 1
        previous_title = title
        span = re.match(r"lines (\d+)-(\d+)", location)
        section = max((line for line in headings if span and line <= int(span[1])), default=0)
        result.append({"title": title, "location": location, "text": text,
                       "group": section if raw_text is not None else group,
                       "code": bool(re.search(r"^\s*(`{3}|~{3})", text, re.M)),
                       "record_type": kind})
        lines = text.splitlines()
        result[-1]["complete_code"] = all(
            token.map[1] - token.map[0] >= 2 and bool(re.fullmatch(
                re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}",
                lines[token.map[1] - 1].strip()))
            for token in MarkdownIt().parse(text) if token.type == "fence")
    return result


def reconstruct_context(query, matches, sources, tokenizer, limit=MAX_CONTEXT_TOKENS):
    """Expand only supported anchors, never across files, headings, or revisions."""
    if not procedure_query(query):
        return matches
    result, seen = [], set()
    for match in matches:
        if match.get("record_type", "faq") != "passage":
            result.append(match)
            continue
        key = (match.get("collection"), match.get("source_path"))
        source = sources.get(key, {})
        blocks = source.get("blocks", [])
        index = match.get("block_index")
        if (not isinstance(index, int) or not 0 <= index < len(blocks)
                or source.get("version") != match.get("version")
                or source.get("source_revision") != match.get("source_revision")):
            # Old indexes cannot reconstruct complete procedures safely.
            result.append({**match, "context_incomplete": True,
                           "matched_answer": "Complete procedure context is unavailable. "
                           "Reimport this source before using it for code or procedure instructions."})
            continue
        group = blocks[index]["group"]
        start = index
        while start > 0 and blocks[start - 1]["group"] == group:
            start -= 1
        end = index + 1
        # Preserve the remaining steps too, including examples sharing setup.
        while end < len(blocks) and blocks[end]["group"] == group:
            end += 1
        identity = (key, source.get("version"), source.get("source_revision"), start, end)
        if identity in seen:
            continue
        seen.add(identity)
        body = "\n\n".join(block["text"] for block in blocks[start:end])
        encoded = tokenizer(body, add_special_tokens=False, truncation=False, verbose=False)
        size = len(encoded.get("input_ids", encoded.get("offset_mapping", [])))
        locations = [block["location"] for block in blocks[start:end]]
        spans = [re.fullmatch(r"lines (\d+)-(\d+)", location) for location in locations]
        location = (f"lines {spans[0][1]}-{spans[-1][2]}" if all(spans)
                    else "; ".join(dict.fromkeys(locations)))
        expanded = {**match, "location": location, "context_reconstructed": True,
                    "context_blocks": end - start, "context_tokens": size,
                    "required_setup": [block["text"] for block in blocks[start:end]
                                       if not block["code"] and re.search(
                                           r"\b(before|must|required|prerequisites?|do not|ensure)\b",
                                           block["text"], re.I)],
                    "matched_answer": body}
        if any(not block.get("complete_code", True) for block in blocks[start:end]):
            expanded.update(context_incomplete=True, matched_answer=
                            "The source contains an incomplete code block. "
                            "Open the cited source to review it; no partial code is supplied.")
        elif size > limit:
            expanded.update(context_incomplete=True, matched_answer=
                            "Complete procedure context exceeds the local reconstruction limit. "
                            "Open the cited source for the full procedure; no partial code is supplied.")
        result.append(expanded)
    # Keep the highest-ranked anchor's position when a later anchor adds context.
    merged = []
    for item in result:
        for index, previous in enumerate(merged):
            if (not item.get("context_reconstructed") or item.get("context_incomplete")
                    or not previous.get("context_reconstructed") or previous.get("context_incomplete")
                    or item.get("source_path") != previous.get("source_path")
                    or item.get("collection") != previous.get("collection")):
                continue
            if item["matched_answer"] in previous["matched_answer"]:
                break
            if previous["matched_answer"] in item["matched_answer"]:
                merged[index] = {**previous, **{key: item[key] for key in (
                    "matched_answer", "location", "context_blocks", "context_tokens", "required_setup")}}
                break
        else:
            merged.append(item)
    return merged
