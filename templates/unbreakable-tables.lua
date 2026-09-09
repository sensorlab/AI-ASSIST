-- Render every table as an unbreakable float instead of a longtable.
--
-- Pandoc always emits longtable for pipe tables, and longtable is designed to split across
-- pages: a twelve-row table landing near a page boundary is silently cut in half, with its
-- header repeated on the next page. These reports are read as tables rather than scanned, so
-- splitting one costs more than the whitespace [H] leaves behind. Requires the float package,
-- loaded in templates/report-tables.tex.
--
-- The rewrite is textual because pandoc has no "don't use longtable" option: the table is
-- written to LaTeX as usual, then the longtable scaffolding is removed. That scaffolding
-- prints the header twice, once in the \endfirsthead block and once in the \endhead block,
-- so the first copy is deleted rather than both markers simply being stripped.
function Table(tbl)
  local tex = pandoc.write(pandoc.Pandoc({ tbl }), "latex")
  tex = tex:gsub("\\begin{longtable}%[%]", "\\begin{tabular}")
  tex = tex:gsub("\\end{longtable}", "\\end{tabular}")
  -- Drop the repeated first-head block, keeping the one that \endhead terminates.
  tex = tex:gsub("\\toprule.-\\endfirsthead%s*", "", 1)
  tex = tex:gsub("\\endhead%s*", "")
  -- longtable puts its bottom rule in a footer block; a tabular needs it at the end instead.
  tex = tex:gsub("\\bottomrule%s*\\noalign{}%s*\\endlastfoot%s*", "")
  tex = tex:gsub("\\end{tabular}", "\\bottomrule\n\\end{tabular}")
  return pandoc.RawBlock("latex", "\\begin{table}[H]\n\\centering\n" .. tex .. "\\end{table}")
end
