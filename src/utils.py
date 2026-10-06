"File with not really used for now functionalites"
import re

class MetaDataCleaner:
    METADATA_RE = re.compile(r'\|[^|]*\||"(?:[^"]|"")*"|(?m:^[ \t]*;[^\n]*\n?)|;[^\n]*|\(\s*set-info\b')
    # Inside a (set-info ...): the same literals and comments, so parens in them don't count.
    SEXP_RE = re.compile(r'\|[^|]*\||"(?:[^"]|"")*"|;[^\n]*|[()]')
    # Rest of a removed set-info's line: spaces, a trailing comment, the newline.
    LINE_END_RE = re.compile(r"[ \t]*(?:;[^\n]*)?\r?\n?")

    @classmethod
    def strip(cls, source: str) -> str:
        # AI generated metadata stripper, wouldn't trust it much
        last = max(source.rfind(";"), source.rfind("set-info"))
        out: list[str] = []
        pos = 0
        while pos <= last and (m := cls.METADATA_RE.search(source, pos)) is not None:
            tok = m.group()
            if tok[0] in '|"':
                out.append(source[pos:m.end()])
                pos = m.end()
                continue
            out.append(source[pos:m.start()])
            pos = m.end()
            if tok.lstrip()[0] == ";":
                continue
            depth = 1   # inside (set-info: skip to its matching close paren
            for t in cls.SEXP_RE.finditer(source, pos):
                depth += {"(": 1, ")": -1}.get(t.group(), 0)
                if depth == 0:
                    pos = t.end()
                    break
            else:
                pos = len(source)
            pos = cls.LINE_END_RE.match(source, pos).end()
        out.append(source[pos:])
        return "".join(out).strip() + "\n"
