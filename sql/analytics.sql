-- =====================================================================
-- Population health analytics on CDC BRFSS survey responses (DuckDB)
-- Table `survey` = raw columns + hd (0/1) + age_mid + bmi_class
-- =====================================================================

-- name: overview
SELECT COUNT(*) AS respondents, SUM(hd) AS heart_disease_cases,
       ROUND(100.0 * AVG(hd), 2) AS prevalence_pct,
       ROUND(AVG(BMI), 1) AS avg_bmi,
       ROUND(100.0 * AVG(CASE WHEN Smoking = 'Yes' THEN 1 ELSE 0 END), 1) AS smokers_pct
FROM survey;

-- name: prevalence_by_age_sex
SELECT AgeCategory, age_mid, Sex, COUNT(*) AS n, ROUND(100.0 * AVG(hd), 2) AS prevalence_pct
FROM survey GROUP BY 1, 2, 3 ORDER BY age_mid, Sex;

-- name: risk_factor_lift
-- Crude relative risk for each yes/no factor: prevalence if Yes / prevalence if No.
-- (Crude = not adjusted for age; see the adjusted odds ratios for the honest picture.)
WITH f AS (
    SELECT 'Stroke' AS factor, Stroke = 'Yes' AS exposed, hd FROM survey UNION ALL
    SELECT 'Kidney disease', KidneyDisease = 'Yes', hd FROM survey UNION ALL
    SELECT 'Difficulty walking', DiffWalking = 'Yes', hd FROM survey UNION ALL
    SELECT 'Diabetes', Diabetic = 'Yes', hd FROM survey UNION ALL
    SELECT 'Skin cancer', SkinCancer = 'Yes', hd FROM survey UNION ALL
    SELECT 'Smoking', Smoking = 'Yes', hd FROM survey UNION ALL
    SELECT 'Asthma', Asthma = 'Yes', hd FROM survey UNION ALL
    SELECT 'Physically inactive', PhysicalActivity = 'No', hd FROM survey UNION ALL
    SELECT 'Heavy drinking', AlcoholDrinking = 'Yes', hd FROM survey
)
SELECT factor,
       ROUND(100.0 * AVG(exposed::INT), 1)                          AS exposed_share_pct,
       ROUND(100.0 * AVG(hd) FILTER (WHERE exposed), 2)             AS prev_if_yes_pct,
       ROUND(100.0 * AVG(hd) FILTER (WHERE NOT exposed), 2)         AS prev_if_no_pct,
       ROUND(AVG(hd) FILTER (WHERE exposed) / AVG(hd) FILTER (WHERE NOT exposed), 2) AS relative_risk
FROM f GROUP BY factor ORDER BY relative_risk DESC;

-- name: gen_health
SELECT GenHealth, COUNT(*) AS n, ROUND(100.0 * AVG(hd), 2) AS prevalence_pct,
       ROUND(AVG(age_mid), 1) AS avg_age
FROM survey GROUP BY 1
ORDER BY CASE GenHealth WHEN 'Excellent' THEN 0 WHEN 'Very good' THEN 1 WHEN 'Good' THEN 2 WHEN 'Fair' THEN 3 ELSE 4 END;

-- name: bmi_classes
SELECT bmi_class, COUNT(*) AS n, ROUND(100.0 * AVG(hd), 2) AS prevalence_pct
FROM survey GROUP BY 1 ORDER BY MIN(BMI);

-- name: comorbidity_burden
-- How does risk stack up as major conditions accumulate?
WITH c AS (
    SELECT hd, (Stroke = 'Yes')::INT + (Diabetic = 'Yes')::INT + (KidneyDisease = 'Yes')::INT
              + (Smoking = 'Yes')::INT + (DiffWalking = 'Yes')::INT AS n_conditions
    FROM survey
)
SELECT LEAST(n_conditions, 3) AS conditions, COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS population_pct,
       ROUND(100.0 * AVG(hd), 2) AS prevalence_pct
FROM c GROUP BY 1 ORDER BY 1;

-- name: lifestyle_matrix
SELECT Smoking, PhysicalActivity, COUNT(*) AS n, ROUND(100.0 * AVG(hd), 2) AS prevalence_pct
FROM survey GROUP BY 1, 2 ORDER BY 1, 2;

-- name: sleep_bands
SELECT CASE WHEN SleepTime <= 5 THEN '<=5 h' WHEN SleepTime <= 6 THEN '6 h' WHEN SleepTime <= 8 THEN '7-8 h'
            WHEN SleepTime <= 9 THEN '9 h' ELSE '10+ h' END AS sleep, MIN(SleepTime) AS k,
       COUNT(*) AS n, ROUND(100.0 * AVG(hd), 2) AS prevalence_pct
FROM survey GROUP BY 1 ORDER BY k;

-- name: race_prevalence
SELECT Race, COUNT(*) AS n, ROUND(100.0 * AVG(hd), 2) AS prevalence_pct, ROUND(AVG(age_mid), 1) AS avg_age
FROM survey GROUP BY 1 ORDER BY n DESC;
