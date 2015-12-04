-- Coursera: University of Washington: DataScience: Assignment 2

-- Problem 1
--  Part a
/*
SELECT count(*) FROM (
    SELECT * FROM frequency
        WHERE docid = "10398_txt_earn"
) x;
*/

--  Part b
/*
SELECT count(*) FROM (
    SELECT term FROM frequency
        WHERE docid = "10398_txt_earn" AND
                count = 1
) x;
*/

--  Part c
/*
SELECT count(*) FROM (
    SELECT term FROM frequency
        WHERE docid = "10398_txt_earn" AND
                count = 1
    UNION
    SELECT term FROM frequency
        WHERE docid = "925_txt_trade" AND
                count = 1    
) x;
*/

--  Part d
/*
SELECT count(*) FROM (
    SELECT DISTINCT docid FROM frequency
        WHERE term = "law" OR term = "legal"
) x;
*/

--  Part e
/* Incorrect
SELECT count(*) FROM (
    SELECT docid, sum(count) AS totaltermcount FROM frequency
        GROUP BY docid
        HAVING totaltermcount > 300
) x;
*/

/*
SELECT count(*) FROM (
    SELECT docid, count(*) AS termcount FROM frequency
        GROUP BY docid
        HAVING termcount > 300
) x;
*/

--  Part f
/*
SELECT count(*) FROM (
    SELECT DISTINCT docid FROM frequency
        WHERE term = "transactions"
    INTERSECT     
    SELECT DISTINCT docid FROM frequency
        WHERE term = "world"    
) x;
*/

--  Part g
/*
-- SELECT ABrow_num, ABcol_num, SUM(ABvalue)
SELECT SUM(ABvalue)
    FROM (
        SELECT  A.row_num AS ABrow_num, A.col_num, B.row_num, B.col_num AS ABcol_num, 
                A.value, B.value, A.value * B.value AS ABvalue
            FROM A, B
            WHERE   A.row_num = 2           AND
                    A.col_num = B.row_num   AND
                    B.col_num = 3  
);
*/

--  Part h
/*
SELECT docid, term, count
    FROM frequency AS doc_col
    WHERE docid = "10080_txt_crude"
;

SELECT doc_col.docid, doc_col.term, doc_col.count
    FROM frequency AS doc_col
    WHERE docid = "10080_txt_crude"
;

SELECT  
        doc_row.docid, doc_row.term, doc_row.count,
        doc_col.docid, doc_col.term, doc_col.count,
        doc_row.count * doc_col.count AS similarity_term 
    FROM frequency AS doc_row, frequency AS doc_col
    WHERE   doc_row.docid = "10080_txt_crude"   AND
            doc_col.docid = "17035_txt_earn"    AND
            doc_row.term  = doc_col.term
;
*/

/*
SELECT  sum(similarity_term) AS similarity_doc 
    FROM (
        SELECT  
                doc_row.docid, doc_row.term, doc_row.count,
                doc_col.docid, doc_col.term, doc_col.count,
                doc_row.count * doc_col.count AS similarity_term 
            FROM frequency AS doc_row, frequency AS doc_col
            WHERE   doc_row.docid = "10080_txt_crude"   AND
                    doc_col.docid = "17035_txt_earn"    AND
                    doc_row.term  = doc_col.term
);
*/

--  Part i
/*
SELECT  row_docid, col_docid, sum(similarity_term) AS similarity_doc 
    FROM (
        SELECT  
                doc_row.docid AS row_docid, doc_row.term, doc_row.count,
                doc_col.docid AS col_docid, doc_col.term, doc_col.count,
                doc_row.count * doc_col.count AS similarity_term 
            FROM frequency AS doc_row, frequency AS doc_col
            WHERE   doc_row.docid = "10080_txt_crude"   AND
                    doc_col.docid = "17035_txt_earn"    AND
                    doc_row.term  = doc_col.term
);

SELECT  row_docid, col_docid, sum(similarity_term) AS similarity_doc 
    FROM (
        SELECT  
                doc_row.docid AS row_docid, doc_row.term, doc_row.count,
                doc_col.docid AS col_docid, doc_col.term, doc_col.count,
                doc_row.count * doc_col.count AS similarity_term 
            FROM frequency AS doc_row, frequency AS doc_col
            WHERE   doc_row.docid = "10080_txt_crude"   AND
                    doc_col.docid = "17035_txt_earn"    AND
                    doc_row.term  = doc_col.term
    )
    GROUP BY row_docid, col_docid
    ORDER BY similarity_doc    
;

SELECT  row_docid, col_docid, sum(similarity_term) AS similarity_doc 
    FROM (
        SELECT  
                doc_row.docid AS row_docid, doc_row.term, doc_row.count,
                doc_col.docid AS col_docid, doc_col.term, doc_col.count,
                doc_row.count * doc_col.count AS similarity_term 
            FROM frequency AS doc_row, frequency AS doc_col
            WHERE   doc_row.docid = "10080_txt_crude"   AND
                    --doc_col.docid = "17035_txt_earn"    AND
                    doc_row.term  = doc_col.term
    )
    GROUP BY row_docid, col_docid
    ORDER BY similarity_doc    
;

SELECT  row_docid, col_docid, sum(similarity_term) AS similarity_doc 
    FROM (
        SELECT  
                doc_row.docid AS row_docid, doc_row.term, doc_row.count,
                doc_col.docid AS col_docid, doc_col.term, doc_col.count,
                doc_row.count * doc_col.count AS similarity_term 
            FROM    (
                        SELECT "q" as docid, "washington"   as term, 1 as count
                        UNION
                        SELECT "q" as docid, "taxes"        as term, 1 as count
                        UNION
                        SELECT "q" as docid, "treasury"     as term, 1 as count
                    )           AS doc_row, 
                    frequency   AS doc_col
            WHERE   doc_row.docid = "q"   AND
                    doc_row.term  = doc_col.term
    )
    GROUP BY row_docid, col_docid
    ORDER BY similarity_doc    
;
*/

SELECT  sum(similarity_term) AS similarity_doc 
    FROM (
        SELECT  
                doc_row.docid AS row_docid, doc_row.term, doc_row.count,
                doc_col.docid AS col_docid, doc_col.term, doc_col.count,
                doc_row.count * doc_col.count AS similarity_term 
            FROM    (
                        SELECT "q" as docid, "washington"   as term, 1 as count
                        UNION
                        SELECT "q" as docid, "taxes"        as term, 1 as count
                        UNION
                        SELECT "q" as docid, "treasury"     as term, 1 as count
                    )           AS doc_row, 
                    frequency   AS doc_col
            WHERE   doc_row.docid = "q"   AND
                    doc_row.term  = doc_col.term
    )
    GROUP BY row_docid, col_docid
    ORDER BY similarity_doc DESC
    LIMIT 1    
;
