-- Let LaTeX size table columns naturally instead of pandoc's proportional p{} widths,
-- which wrap short cells like "recorded location" into a ragged column.
function Table(tbl)
  for _, spec in ipairs(tbl.colspecs) do
    spec[2] = nil
  end
  return tbl
end
