SELECT
  * 
FROM
  (submissions s1 JOIN images i1 JOIN projects)
  LEFT JOIN (submissions s2 JOIN images i2) ON
    i1.id = i2.id AND s1.created_at > s2.created_at
WHERE
  projects.id IN (<project_ids>)
  AND m2.id IS NULL
