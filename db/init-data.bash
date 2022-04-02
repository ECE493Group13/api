#!/usr/bin/env bash

DB_URL=$1

db_exec() {
    psql $DB_URL -c "$1"
}

db_exec 'DROP SCHEMA IF EXISTS docs CASCADE'
db_exec 'CREATE SCHEMA docs'
# index data sets role to "roger"
db_exec 'DROP ROLE IF EXISTS roger'
db_exec 'CREATE ROLE roger'

echo 'Unzip metadata (~1.5 minutes)'
unzip -p ~/data/GeneralIndex.info.0/doc_info_0.sql.zip | tqdm --dynamic-ncols --smoothing 0 --unit-scale --total 7320256 | psql $DB_URL

echo 'Remove non-unique metadata dkeys'

# Remove non-unique documents (~1%) and add primary key
db_exec 'DELETE FROM docs.doc_meta_0 WHERE dkey NOT IN (SELECT dkey FROM docs.doc_meta_0 GROUP BY dkey HAVING count(*) = 1)'
db_exec 'ALTER TABLE docs.doc_meta_0 ADD PRIMARY KEY (dkey)'

echo 'Unzip keywords (~40 minutes)'
unzip -p ~/data/GeneralIndex.keywords.0/doc_keywords_0.sql.zip | tqdm --dynamic-ncols --smoothing 0 --unit-scale --total 1240042165 | psql $DB_URL

echo 'Unzip ngrams (~12 hours / 12 = 1 hour)'
unzip -p ~/data/GeneralIndex.ngrams.0/doc_ngrams_0.sql.zip | head -n 1850415729 | tqdm --dynamic-ncols --smoothing 0 --unit-scale --total 1850415729 | psql $DB_URL

echo 'Create indices for ngrams'

db_exec 'CREATE INDEX doc_ngrams_0_dkey_idx ON docs.doc_ngrams_0 USING btree (dkey)'
db_exec 'CREATE INDEX doc_ngrams_0_ngram_lc_idx ON docs.doc_ngrams_0 USING btree (ngram_lc)'

echo 'Analyze'
db_exec 'alter system set default_statistics_target=500'
db_exec 'analyze'

db_exec 'GRANT ALL ON SCHEMA docs TO public'
