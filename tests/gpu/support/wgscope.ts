// TSL branch-scope trap (B19, B25): a node first built inside one If/ElseIf branch is declared at the top of main()
// but assigned only in that branch; reads anywhere else see 0. Finds such reads of TSL temps (nodeVarN) in WGSL.
// Heuristic: a read counts as safe when an assignment in the same or an enclosing scope comes first, or both arms
// of an if/else directly inside an enclosing scope assign it (TSL's select / cond pattern). Loop bodies count as
// their enclosing scope.
export interface ScopeViolation { fn: string; v: string; line: number; assignedIn: string[]; readIn: string }

export function checkWgslScopes(code: string): ScopeViolation[] {
  const out: ScopeViolation[] = [];
  // split into functions
  const fnRe = /fn\s+(\w+)\s*\(/g;
  let m: RegExpExecArray | null; const starts: [number, string][] = [];
  while ((m = fnRe.exec(code))) starts.push([m.index, m[1]!]);
  for (let f = 0; f < starts.length; f++) {
    const body = code.slice(starts[f]![0], f + 1 < starts.length ? starts[f + 1]![0] : code.length);
    const stack: number[] = [0]; let next = 1; const path = () => stack.filter((k) => k >= 0).join('/'); // -1 = loop body, same scope as its parent
    const assigned = new Map<string, string[]>(); // var -> list of scope paths where assigned
    const tok = /\{|\}|(nodeVar\d+)(\s*(?:=(?!=)|\+=|-=|\*=|\/=))?/g;
    let t: RegExpExecArray | null;
    const lineOf = (i: number) => body.slice(0, i).split('\n').length;
    while ((t = tok.exec(body))) {
      // a for-loop body runs (TSL loops have fixed, positive counts): its assignments dominate what follows
      if (t[0] === '{') { const loop = /for\s*\([^{]*\)\s*$/.test(body.slice(Math.max(0, t.index - 120), t.index)); stack.push(loop ? -1 : next++); continue; }
      if (t[0] === '}') { stack.pop(); continue; }
      const v = t[1]!;
      // declarations "var nodeVarN : T;" are at fn top: skip
      const before = body.slice(Math.max(0, t.index - 4), t.index);
      if (/var\s$/.test(before)) continue;
      const p = path();
      if (t[2] && !/[+\-*/]=/.test(t[2])) {
        // plain assignment: RHS reads happen before; record assignment
        (assigned.get(v) ?? assigned.set(v, []).get(v)!).push(p);
        continue;
      }
      const as = assigned.get(v) ?? [];
      // dominated by an earlier assignment in this scope or an ancestor, or by both arms of an if/else
      // directly inside it (TSL's value-returning select / cond pattern)
      const parts = p.split('/');
      const anc = parts.map((_, i) => parts.slice(0, i + 1).join('/')); // p and its ancestors
      const kidsOf = (q: string) => new Set(as.filter((a) => a.startsWith(q + '/') && !a.slice(q.length + 1).includes('/') && !p.startsWith(a)));
      const ok = as.some((a) => anc.includes(a)) || anc.some((q) => kidsOf(q).size >= 2);
      if (!ok) out.push({ fn: starts[f]![1], v, line: lineOf(t.index), assignedIn: as.slice(0, 3), readIn: p });
    }
  }
  return out;
}
