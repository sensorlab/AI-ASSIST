-- Let LaTeX size table columns naturally instead of using pandoc's proportional p{} widths.
--
-- Pandoc infers column widths from the Markdown source, which for these reports produced
-- fixed-width paragraph columns that wrapped short cells: "recorded location" came out broken
-- across three lines, and the wrapping cost more vertical space than the table saved.
--
-- The trade-off is that nothing wraps any more, so a table holding a very long unbreakable
-- token runs into the margin. Keep such cells short in the source instead: the artifact
-- listings name files rather than full repository paths for exactly this reason.
function Table(tbl)
  for _, spec in ipairs(tbl.colspecs) do
    spec[2] = nil
  end
  return tbl
end
