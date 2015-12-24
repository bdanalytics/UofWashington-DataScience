# Seaflow: pop classification:: seaflow_base
bdanalytics  

**  **    
**Date: (Wed) Dec 23, 2015**    

# Introduction:  

Data: SeaFlow environmental flow cytometry
Source: Armbrust Lab at the University of Washington
    Training:   seaflow_21min.csv  
    New:        <obsNewFileName>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

Summary of key steps & error improvement stats:

### Prediction Accuracy Enhancement Options:
- transform.data chunk:
    - derive features from multiple features
    
- manage.missing.data chunk:
    - Not fill missing vars
    - Fill missing numerics with a different algorithm
    - Fill missing chars with data based on clusters 
    
- extract.features chunk:
    - Text variables: move to date extraction chunk ???
        - Mine acronyms
        - Mine places

- Review set_global_options chunk after features are finalized

### ![](<filename>.png)

## Potential next steps include:
- Organization:
    - Categorize by chunk
    - Priority criteria:
        0. Ease of change
        1. Impacts report
        2. Cleans innards
        3. Bug report
        
- all chunks:
    - at chunk-end rm(!glb_<var>)
    
- manage.missing.data chunk:
    - cleaner way to manage re-splitting of training vs. new entity

- extract.features chunk:
    - Add n-grams for glbFeatsText
        - "RTextTools", "tau", "RWeka", and "textcat" packages
    - Convert user-specified mutate code to config specs
    
- fit.models chunk:
    - Prediction accuracy scatter graph:
    -   Add tiles (raw vs. PCA)
    -   Use shiny for drop-down of "important" features
    -   Use plot.ly for interactive plots ?
    
    - Change .fit suffix of model metrics to .mdl if it's data independent (e.g. AIC, Adj.R.Squared - is it truly data independent ?, etc.)
    - create a custom model for rpart that has minbucket as a tuning parameter
    - varImp for randomForest crashes in caret version:6.0.41 -> submit bug report

- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- Skip trControl.method="cv" for dummy classifier ?

- fit.all.training chunk:
    - myplot_prediction_classification: displays 'x' instead of '+' when there are no prediction errors 
- Compare glb_sel_mdl vs. glb_fin_mdl:
    - varImp
    - Prediction differences (shd be minimal ?)

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list = ls())
set.seed(12345)
options(stringsAsFactors = FALSE)
source("~/Dropbox/datascience/R/myscript.R")
source("~/Dropbox/datascience/R/mydsutils.R")
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
source("~/Dropbox/datascience/R/myplclust.R")
source("~/Dropbox/datascience/R/mytm.R")
# Gather all package requirements here
suppressPackageStartupMessages(require(doMC))
registerDoMC(6) # # of cores on machine - 2
suppressPackageStartupMessages(require(caret))
require(plyr)
```

```
## Loading required package: plyr
```

```r
require(dplyr)
```

```
## Loading required package: dplyr
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:plyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
#source("dbgcaret.R")
#packageVersion("snow")
#require(sos); findFn("cosine", maxPages=2, sortby="MaxScore")

# Analysis control global variables
# Inputs
#   url/name = "seaflow_21min.csv"; sep = choose from c(NULL, "\t")
glbObsTrnFile <- list(name = "seaflow_21min.csv") 
glbObsNewFile <- list(url = "<obsNewFileName>")
glbInpMerge <- NULL #: default
#     list(fnames = c("<fname1>", "<fname2>")) # files will be concatenated

glb_is_separate_newobs_dataset <- FALSE    # or TRUE
    glb_split_entity_newobs_datasets <- TRUE   # FALSE not supported - use "copy" for glb_split_newdata_method # select from c(FALSE, TRUE)
    glb_split_newdata_method <- "sample" # select from c(NULL, "condition", "sample", "copy")
    glb_split_newdata_condition <- NULL # or "is.na(<var>)"; "<var> <condition_operator> <value>"
    glb_split_newdata_size_ratio <- 0.5               # > 0 & < 1
    glb_split_sample.seed <- 123               # or any integer

glbObsDropCondition <- NULL # : default
#            "<condition>" # use | & ; NOT || &&
#parse(text=glbObsDropCondition)
#subset(glbObsAll, .grpid %in% c(31))
    
glb_obs_repartition_train_condition <- NULL # : default
#    "<condition>" 

glb_max_fitobs <- NULL # or any integer
                         
glb_is_regression <- FALSE; glb_is_classification <- !glb_is_regression; 
    glb_is_binomial <- FALSE # or TRUE or FALSE

glb_rsp_var_raw <- "pop"

# for classification, the response variable has to be a factor
glb_rsp_var <- "pop.fctr" # glb_rsp_var_raw # or "pop.fctr"

# if the response factor is based on numbers/logicals e.g (0/1 OR TRUE/FALSE vs. "A"/"B"), 
#   or contains spaces (e.g. "Not in Labor Force")
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- # NULL 
function(raw) {
#     return(raw ^ 0.5)
#     return(log(raw))
#     return(log(1 + raw))
#     return(log10(raw)) 
#     return(exp(-raw / 2))
#     ret_vals <- rep_len(NA, length(raw)); ret_vals[!is.na(raw)] <- ifelse(raw[!is.na(raw)] == 1, "Y", "N"); return(relevel(as.factor(ret_vals), ref="N"))
#     #as.factor(paste0("B", raw))
    as.factor(gsub(" ", "\\.", raw))    
}

#if glb_rsp_var_raw is numeric:
#print(summary(glbObsAll[, glb_rsp_var_raw]))
#glb_map_rsp_raw_to_var(tst <- c(NA, as.numeric(summary(glbObsAll[, glb_rsp_var_raw])))) 

#if glb_rsp_var_raw is character:
#print(table(glbObsAll[, glb_rsp_var_raw]))
#glb_map_rsp_raw_to_var(tst <- c(NA, names(table(glbObsAll[, glb_rsp_var_raw])))) 


glb_map_rsp_var_to_raw <- NULL 
# function(var) {
#     return(var ^ 2.0)
#     return(exp(var))
#     return(10 ^ var) 
#     return(-log(var) * 2)
#     as.numeric(var)
#     gsub("\\.", " ", levels(var)[as.numeric(var)])
#     c("<=50K", " >50K")[as.numeric(var)]
#     c(FALSE, TRUE)[as.numeric(var)]
# }
# glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(tst))

if ((glb_rsp_var != glb_rsp_var_raw) && is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

# List info gathered for various columns
# <col_name>:   <description>; <notes>
# file_id: The data arrives in files, where each file represents a three-minute window; this field represents which file the data came from. The number is ordered by time, but is otherwise not significant.
# 
# · time: This is an integer representing the time the particle passed through the instrument. Many particles may arrive at the same time; time is not a key for this relation.
# 
# · cell_id: A unique identifier for each cell WITHIN a file. (file_id, cell_id) is a key for this relation.
# 
# · d1, d2: Intensity of light at the two main sensors, oriented perpendicularly. These sensors are primarily used to determine whether the particles are properly centered in the stream. Used primarily in preprocesssing; they are unlikely to be useful for classification.
# 
# · fsc_small, fsc_perp, fsc_big: Forward scatter small, perpendicular, and big. These values help distingish different sizes of particles.
# 
# · pe: A measurement of phycoerythrin fluorescence, which is related to the wavelength associated with an orange color in microorganisms
# 
# · chl_small, chl_big: Measurements related to the wavelength of light corresponding to chlorophyll.
# 
# · pop: This is the class label assigned by the clustering mechanism used in the production system. It can be considered "ground truth" for the purposes of the assignment, but note that there are particles that cannot be unambiguously classified, so you should not aim for 100% accuracy. The values in this column are crypto, nano, pico, synecho, and ultra

# If multiple vars are parts of id, consider concatenating them to create one id var
# If glb_id_var == NULL, ".rownames <- as.numeric(row.names())" is the default

# User-specified exclusions
glbFeatsExclude <- c(NULL
#   Feats that shd be excluded due to known causation by prediction variable
# , "<feat1", "<feat2>"
#   Feats that are linear combinations (alias in glm)
#   Feature-engineering phase -> start by excluding all features except id & category & work each one in
                    ) 
if (glb_rsp_var_raw != glb_rsp_var)
    glbFeatsExclude <- union(glbFeatsExclude, glb_rsp_var_raw)                    

glbFeatsInteractionOnly <- list()
#glbFeatsInteractionOnly[["<child_feat>"]] <- "<parent_feat>"

# currently does not handle more than 1 column; consider concatenating multiple columns
glb_id_var <- NULL # choose from c(NULL : default, "<id_feat>") 
glbFeatsCategory <- NULL # choose from c(NULL : default, "<category_feat>")

glb_drop_vars <- c(NULL
                # , "<feat1>", "<feat2>"
                )

glb_map_vars <- NULL # or c("<var1>", "<var2>")
glb_map_urls <- list();
# glb_map_urls[["<var1>"]] <- "<var1.url>"

glb_assign_pairs_lst <- NULL; 
# glb_assign_pairs_lst[["<var1>"]] <- list(from=c(NA),
#                                            to=c("NA.my"))
glb_assign_vars <- names(glb_assign_pairs_lst)

# Derived features; Use this mechanism to cleanse data ??? Cons: Data duplication ???
glbFeatsDerive <- list();

# glbFeatsDerive[["<feat.my.sfx>"]] <- list(
#     mapfn = function(<arg1>, <arg2>) { return(function(<arg1>, <arg2>)) } 
#   , args = c("<arg1>", "<arg2>"))

    # character
#     mapfn = function(Week) { return(substr(Week, 1, 10)) }

#     mapfn = function(descriptor) { return(plyr::revalue(descriptor, c(
#         "ABANDONED BUILDING"  = "OTHER",
#         "**"                  = "**"
#                                           ))) }

#     mapfn = function(description) { mod_raw <- description;
    # This is here because it does not work if it's in txt_map_filename
#         mod_raw <- gsub(paste0(c("\n", "\211", "\235", "\317", "\333"), collapse = "|"), " ", mod_raw)
    # Don't parse for "." because of ".com"; use customized gsub for that text
#         mod_raw <- gsub("(\\w)(!|\\*|,|-|/)(\\w)", "\\1\\2 \\3", mod_raw);
#         return(mod_raw) }
#print(mod_raw <- grep("&#034;", glbObsAll[, txt_var], value = TRUE)) 
#print(mod_raw <- glbObsAll[c(88,187,280,1040,1098), txt_var])
#print(mod_raw <- glbObsAll[sel_obs(list(descr.my.contains="\\bdoes( +)not\\b")), glbFeatsText])
#print(mod_raw <- glbObsAll[sel_obs(list(descr.my.contains="\\bipad [[:digit:]]\\b")), glbFeatsText][01:10])
#print(mod_raw <- glbObsAll[sel_obs(list(descr.my.contains="pad mini")), glbFeatsText][11:20])
#print(mod_raw <- glbObsAll[sel_obs(list(descr.my.contains="pad mini")), glbFeatsText][21:30])
#print(mod_raw <- glbObsAll[sel_obs(list(descr.my.contains="pad mini")), glbFeatsText][31:40])
#glbObsAll[which(glb_post_stop_words_terms_mtrx_lst[[txt_var]][, subset(glb_post_stop_words_terms_df_lst[[txt_var]], term %in% c("conditionminimal"))$pos] > 0), "description"]

    # numeric
# Create feature based on record position/id in data   
glbFeatsDerive[[".pos"]] <- list(
    mapfn = function(.rnorm) { return(1:length(.rnorm)) }       
    , args = c(".rnorm"))    

# Add logs of numerics that are not distributed normally
#   Derive & keep multiple transformations of the same feature, if normality is hard to achieve with just one transformation
#   Right skew: logp1; sqrt; ^ 1/3; logp1(logp1); log10; exp(-<feat>/constant)
# glbFeatsDerive[["WordCount.log1p"]] <- list(
#     mapfn = function(WordCount) { return(log1p(WordCount)) } 
#   , args = c("WordCount"))
# glbFeatsDerive[["WordCount.root2"]] <- list(
#     mapfn = function(WordCount) { return(WordCount ^ (1/2)) } 
#   , args = c("WordCount"))
# glbFeatsDerive[["WordCount.nexp"]] <- list(
#     mapfn = function(WordCount) { return(exp(-WordCount)) } 
#   , args = c("WordCount"))
#print(summary(glbObsAll$WordCount))
#print(summary(mapfn(glbObsAll$WordCount)))
    
#     mapfn = function(HOSPI.COST) { return(cut(HOSPI.COST, 5, breaks = c(0, 100000, 200000, 300000, 900000), labels = NULL)) }     
#     mapfn = function(Rasmussen)  { return(ifelse(sign(Rasmussen) >= 0, 1, 0)) } 
#     mapfn = function(startprice) { return(startprice ^ (1/2)) }       
#     mapfn = function(startprice) { return(log(startprice)) }   
#     mapfn = function(startprice) { return(exp(-startprice / 20)) }
#     mapfn = function(startprice) { return(scale(log(startprice))) }     
#     mapfn = function(startprice) { return(sign(sprice.predict.diff) * (abs(sprice.predict.diff) ^ (1/10))) }        

    # factor      
#     mapfn = function(PropR) { return(as.factor(ifelse(PropR >= 0.5, "Y", "N"))) }
#     mapfn = function(productline, description) { as.factor(gsub(" ", "", productline)) }
#     mapfn = function(purpose) { return(relevel(as.factor(purpose), ref="all_other")) }
#     mapfn = function(raw) { tfr_raw <- as.character(cut(raw, 5)); 
#                             tfr_raw[is.na(tfr_raw)] <- "NA.my";
#                             return(as.factor(tfr_raw)) }
#     mapfn = function(startprice.log10) { return(cut(startprice.log10, 3)) }
#     mapfn = function(startprice.log10) { return(cut(sprice.predict.diff, c(-1000, -100, -10, -1, 0, 1, 10, 100, 1000))) }    

#     , args = c("<arg1>"))
    
    # multiple args    
#     mapfn = function(PTS, oppPTS) { return(PTS - oppPTS) }
#     mapfn = function(startprice.log10.predict, startprice) {
#                  return(spdiff <- (10 ^ startprice.log10.predict) - startprice) } 
#     mapfn = function(productline, description) { as.factor(
#         paste(gsub(" ", "", productline), as.numeric(nchar(description) > 0), sep = "*")) }

# # If glbObsAll is not sorted in the desired manner
#     mapfn=function(Week) { return(coredata(lag(zoo(orderBy(~Week, glbObsAll)$ILI), -2, na.pad=TRUE))) }
#     mapfn=function(ILI) { return(coredata(lag(zoo(ILI), -2, na.pad=TRUE))) }
#     mapfn=function(ILI.2.lag) { return(log(ILI.2.lag)) }

# glbFeatsDerive[["<var1>"]] <- glbFeatsDerive[["<var2>"]]

glb_derive_vars <- names(glbFeatsDerive)

# tst <- "descr.my"; args_lst <- NULL; for (arg in glbFeatsDerive[[tst]]$args) args_lst[[arg]] <- glbObsAll[, arg]; print(head(args_lst[[arg]])); print(head(drv_vals <- do.call(glbFeatsDerive[[tst]]$mapfn, args_lst))); 
# print(which_ix <- which(args_lst[[arg]] == 0.75)); print(drv_vals[which_ix]); 

glbFeatsDateTime <- list()
# glbFeatsDateTime[["<DateTimeFeat>"]] <- 
#     c(format = "%Y-%m-%d %H:%M:%S", timezone = "America/New_York", impute.na = TRUE, 
#       last.ctg = TRUE, poly.ctg = TRUE)

glbFeatsPrice <- NULL # or c("<price_var>")

glbFeatsText <- NULL # c("<txt_var>")   # NULL # 
Sys.setlocale("LC_ALL", "C") # For english
```

```
## [1] "C/C/C/C/C/en_US.UTF-8"
```

```r
# Text Processing Step: custom modifications not present in txt_munge -> use glbFeatsDerive
# Text Processing Step: universal modifications
glb_txt_munge_filenames_pfx <- "<projectId>_mytxt_"

# Text Processing Step: tolower
# Text Processing Step: myreplacePunctuation
# Text Processing Step: removeWords
glb_txt_stop_words <- list()
# Remember to use unstemmed words
if (!is.null(glbFeatsText)) {
    require(tm)

    glb_txt_stop_words[["<txt_var>"]] <- sort(c(NULL    

        # Remove any words from stopwords            
#         , setdiff(myreplacePunctuation(stopwords("english")), c("<keep_wrd1>", <keep_wrd2>"))
                                
        # cor.y.train == NA
#         ,unlist(strsplit(paste(c(NULL
#           ,"<comma-separated-terms>"
#         ), collapse=",")

        # freq == 1; keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>

        # chisq.pval high (e.g. == 1); keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>

        # nzv.freqRatio high (e.g. >= glb_nzv_freqCut); keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>        
                                            ))
}
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txt_var]][grep("^2", glb_post_stem_words_terms_df_lst[[txt_var]]$term), ])
#glbObsAll[glb_post_stem_words_terms_mtrx_lst[[txt_var]][, 6] > 0, glbFeatsText]

# To identify terms with a specific freq
#paste0(sort(subset(glb_post_stop_words_terms_df_lst[[txt_var]], freq == 1)$term), collapse = ",")
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txt_var]], freq <= 2)$term), collapse = ",")

# To identify terms with a specific freq & 
#   are not stemmed together later OR is value of color.fctr (e.g. gold)
#paste0(sort(subset(glb_post_stop_words_terms_df_lst[[txt_var]], (freq == 1) & !(term %in% c("blacked","blemish","blocked","blocks","buying","cables","careful","carefully","changed","changing","chargers","cleanly","cleared","connect","connects","connected","contains","cosmetics","default","defaulting","defective","definitely","describe","described","devices","displays","drop","drops","engravement","excellant","excellently","feels","fix","flawlessly","frame","framing","gentle","gold","guarantee","guarantees","handled","handling","having","install","iphone","iphones","keeped","keeps","known","lights","line","lining","liquid","liquidation","looking","lots","manuals","manufacture","minis","most","mostly","network","networks","noted","opening","operated","performance","performs","person","personalized","photograph","physically","placed","places","powering","pre","previously","products","protection","purchasing","returned","rotate","rotation","running","sales","second","seconds","shipped","shuts","sides","skin","skinned","sticker","storing","thats","theres","touching","unusable","update","updates","upgrade","weeks","wrapped","verified","verify") ))$term), collapse = ",")

#print(subset(glb_post_stem_words_terms_df_lst[[txt_var]], (freq <= 2)))
#glbObsAll[which(terms_mtrx[, 229] > 0), glbFeatsText]

# To identify terms with cor.y == NA
#orderBy(~-freq+term, subset(glb_post_stop_words_terms_df_lst[[txt_var]], is.na(cor.y)))
#paste(sort(subset(glb_post_stop_words_terms_df_lst[[txt_var]], is.na(cor.y))[, "term"]), collapse=",")
#orderBy(~-freq+term, subset(glb_post_stem_words_terms_df_lst[[txt_var]], is.na(cor.y)))

# To identify terms with low cor.y.abs
#head(orderBy(~cor.y.abs+freq+term, subset(glb_post_stem_words_terms_df_lst[[txt_var]], !is.na(cor.y))), 5)

# To identify terms with high chisq.pval
#subset(glb_post_stem_words_terms_df_lst[[txt_var]], chisq.pval > 0.99)
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txt_var]], (chisq.pval > 0.99) & (freq <= 10))$term), collapse=",")
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txt_var]], (chisq.pval > 0.9))$term), collapse=",")
#head(orderBy(~-chisq.pval+freq+term, glb_post_stem_words_terms_df_lst[[txt_var]]), 5)
#glbObsAll[glb_post_stem_words_terms_mtrx_lst[[txt_var]][, 68] > 0, glbFeatsText]
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txt_var]][grep("^m", glb_post_stem_words_terms_df_lst[[txt_var]]$term), ])

# To identify terms with high nzv.freqRatio
#summary(glb_post_stem_words_terms_df_lst[[txt_var]]$nzv.freqRatio)
#paste0(sort(setdiff(subset(glb_post_stem_words_terms_df_lst[[txt_var]], (nzv.freqRatio >= glb_nzv_freqCut) & (freq < 10) & (chisq.pval >= 0.05))$term, c( "128gb","3g","4g","gold","ipad1","ipad3","ipad4","ipadair2","ipadmini2","manufactur","spacegray","sprint","tmobil","verizon","wifion"))), collapse=",")

# To identify obs with a txt term
#tail(orderBy(~-freq+term, glb_post_stop_words_terms_df_lst[[txt_var]]), 20)
#mydspObs(list(descr.my.contains="non"), cols=c("color", "carrier", "cellular", "storage"))
#grep("ever", dimnames(terms_stop_mtrx)$Terms)
#which(terms_stop_mtrx[, grep("ipad", dimnames(terms_stop_mtrx)$Terms)] > 0)
#glbObsAll[which(terms_stop_mtrx[, grep("16", dimnames(terms_stop_mtrx)$Terms)[1]] > 0), c(glbFeatsCategory, "storage", txt_var)]

# To identify whether terms shd be synonyms
#orderBy(~term, glb_post_stop_words_terms_df_lst[[txt_var]][grep("^moder", glb_post_stop_words_terms_df_lst[[txt_var]]$term), ])
# term_row_df <- glb_post_stop_words_terms_df_lst[[txt_var]][grep("^came$", glb_post_stop_words_terms_df_lst[[txt_var]]$term), ]
# 
# cor(glb_post_stop_words_terms_mtrx_lst[[txt_var]][glbObsAll$.lcn == "Fit", term_row_df$pos], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")

# To identify which stopped words are "close" to a txt term
#sort(cluster_vars)

# Text Processing Step: stemDocument
# To identify stemmed txt terms
#glb_post_stop_words_terms_df_lst[[txt_var]][grep("condit", glb_post_stop_words_terms_df_lst[[txt_var]]$term), ]
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txt_var]][grep("^con", glb_post_stem_words_terms_df_lst[[txt_var]]$term), ])
#glbObsAll[which(terms_stem_mtrx[, grep("use", dimnames(terms_stem_mtrx)$Terms)[[1]]] > 0), c(glb_id_var, "productline", txt_var)]
#glbObsAll[which(TfIdf_stem_mtrx[, 191] > 0), c(glb_id_var, glbFeatsCategory, txt_var)]
#which(glbObsAll$UniqueID %in% c(11915, 11926, 12198))

# Text Processing Step: mycombineSynonyms
#   To identify which terms are associated with not -> combine "could not" & "couldn't"
#findAssocs(glb_full_DTM_lst[[txt_var]], "not", 0.05)
#   To identify which synonyms should be combined
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txt_var]][grep("^c", glb_post_stem_words_terms_df_lst[[txt_var]]$term), ])
chk_comb_cor <- function(syn_lst) {
#     cor(terms_stem_mtrx[glbObsAll$.src == "Train", grep("^(damag|dent|ding)$", dimnames(terms_stem_mtrx)[[2]])], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
    print(subset(glb_post_stem_words_terms_df_lst[[txt_var]], term %in% syn_lst$syns))
    print(subset(get_corpus_terms(tm_map(glb_txt_corpus_lst[[txt_var]], mycombineSynonyms, list(syn_lst), lazy=FALSE)), term == syn_lst$word))
#     cor(terms_stop_mtrx[glbObsAll$.src == "Train", grep("^(damage|dent|ding)$", dimnames(terms_stop_mtrx)[[2]])], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
#     cor(rowSums(terms_stop_mtrx[glbObsAll$.src == "Train", grep("^(damage|dent|ding)$", dimnames(terms_stop_mtrx)[[2]])]), glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
}
#chk_comb_cor(syn_lst=list(word="cabl",  syns=c("cabl", "cord")))
#chk_comb_cor(syn_lst=list(word="damag",  syns=c("damag", "dent", "ding")))
#chk_comb_cor(syn_lst=list(word="dent",  syns=c("dent", "ding")))
#chk_comb_cor(syn_lst=list(word="use",  syns=c("use", "usag")))

glb_txt_synonyms <- list()
#glb_txt_synonyms[["<txt_var>"]] <- list(NULL
#     , list(word="<stem1>",  syns=c("<stem1>", "<stem1_2>"))
#                                       )

# options include: "weightTf", "myweightTflog1p", "myweightTfsqrt", "weightTfIdf", "weightBM25"
glb_txt_terms_control <- list(weighting = "weightTfIdf" # : default
                # termFreq selection criteria across obs: tm default: list(global=c(1, Inf))
                    , bounds = list(global = c(1, Inf)) 
                # wordLengths selection criteria: tm default: c(3, Inf)
                    , wordLengths = c(1, Inf) 
                              ) 

glb_txt_cor_var <- glb_rsp_var # : default # or c(<feat>)

# select one from c("union.top.val.cor", "top.cor", "top.val", default: "top.chisq", "sparse")
glbFeatsTextFilter <- "top.chisq" 
glbFeatsTextTermsMax <- rep(10, length(glbFeatsText)) # :default
names(glbFeatsTextTermsMax) <- glbFeatsText

# Text Processing Step: extractAssoc
glbFeatsTextAssocCor <- rep(1, length(glbFeatsText)) # :default 
names(glbFeatsTextAssocCor) <- glbFeatsText

# Remember to use stemmed terms
glb_important_terms <- list()

# Text Processing Step: extractPatterns (ngrams)
glbFeatsTextPatterns <- list()
#glbFeatsTextPatterns[[<txt_var>>]] <- list()
#glbFeatsTextPatterns[[<txt_var>>]] <- c(metropolitan.diary.colon = "Metropolitan Diary:")

# Have to set it even if it is not used
# Properties:
#   numrows(glb_feats_df) << numrows(glbObsFit
#   Select terms that appear in at least 0.2 * O(FP/FN(glbObsOOB)) ???
#       numrows(glbObsOOB) = 1.1 * numrows(glbObsNew) ???
glb_sprs_thresholds <- NULL # or c(<txt_var1> = 0.988, <txt_var2> = 0.970, <txt_var3> = 0.970)

glbFctrMaxUniqVals <- 20 # default: 20
glb_impute_na_data <- FALSE # or TRUE
glb_mice_complete.seed <- 144 # or any integer

glb_cluster <- FALSE # : default or TRUE
glb_cluster.seed <- 189 # or any integer
glb_cluster_entropy_var <- NULL # c(glb_rsp_var, as.factor(cut(glb_rsp_var, 3)), default: NULL)
glbFeatsTextClusterVarsExclude <- FALSE # default FALSE

glb_interaction_only_feats <- NULL # : default or c(<parent_feat> = "<child_feat>")

glb_nzv_freqCut <- 19 # 19 : caret default
glb_nzv_uniqueCut <- 10 # 10 : caret default

glbRFESizes <- list()
#glbRFESizes[["mdlFamily"]] <- c(4, 8, 16, 32, 64, 67, 68, 69) # Accuracy@69/70 = 0.8258

glbObsFitOutliers <- list()
# If outliers.n >= 10; consider concatenation of interaction vars
# glbObsFitOutliers[["<mdlFamily>"]] <- c(NULL
    # is.na(.rstudent)
    # is.na(.dffits)
    # .hatvalues >= 0.99        
    # -38,167,642 < minmax(.rstudent) < 49,649,823    
#     , <comma-separated-<glb_id_var>>
#                                     )
glbObsTrnOutliers <- list()

# influence.measures: car::outlier; rstudent; dffits; hatvalues; dfbeta; dfbetas
#mdlId <- "RFE.X.glm"; obs_df <- fitobs_df
#mdlId <- "Final.glm"; obs_df <- trnobs_df
#mdlId <- "CSM2.X.glm"; obs_df <- fitobs_df
#print(outliers <- car::outlierTest(glb_models_lst[[mdlId]]$finalModel))
#mdlIdFamily <- paste0(head(unlist(str_split(mdlId, "\\.")), -1), collapse="."); obs_df <- dplyr::filter_(obs_df, interp(~(!(var %in% glbObsFitOutliers[[mdlIdFamily]])), var = as.name(glb_id_var))); model_diags_df <- cbind(obs_df, data.frame(.rstudent=stats::rstudent(glb_models_lst[[mdlId]]$finalModel)), data.frame(.dffits=stats::dffits(glb_models_lst[[mdlId]]$finalModel)), data.frame(.hatvalues=stats::hatvalues(glb_models_lst[[mdlId]]$finalModel)));print(summary(model_diags_df[, c(".rstudent",".dffits",".hatvalues")])); table(cut(model_diags_df$.hatvalues, breaks=c(0.00, 0.98, 0.99, 1.00)))

#print(subset(model_diags_df, is.na(.rstudent))[, glb_id_var])
#print(subset(model_diags_df, is.na(.dffits))[, glb_id_var])
#print(model_diags_df[which.min(model_diags_df$.dffits), ])
#print(subset(model_diags_df, .hatvalues > 0.99)[, glb_id_var])
#dffits_df <- merge(dffits_df, outliers_df, by="row.names", all.x=TRUE); row.names(dffits_df) <- dffits_df$Row.names; dffits_df <- subset(dffits_df, select=-Row.names)
#dffits_df <- merge(dffits_df, glbObsFit, by="row.names", all.x=TRUE); row.names(dffits_df) <- dffits_df$Row.names; dffits_df <- subset(dffits_df, select=-Row.names)
#subset(dffits_df, !is.na(.Bonf.p))

#mdlId <- "CSM.X.glm"; vars <- myextract_actual_feats(row.names(orderBy(reformulate(c("-", paste0(mdlId, ".imp"))), myget_feats_imp(glb_models_lst[[mdlId]])))); 
#model_diags_df <- glb_get_predictions(model_diags_df, mdlId, glb_rsp_var)
#obs_ix <- row.names(model_diags_df) %in% names(outliers$rstudent)[1]
#obs_ix <- which(is.na(model_diags_df$.rstudent))
#obs_ix <- which(is.na(model_diags_df$.dffits))
#myplot_parcoord(obs_df=model_diags_df[, c(glb_id_var, glbFeatsCategory, ".rstudent", ".dffits", ".hatvalues", glb_rsp_var, paste0(glb_rsp_var, mdlId), vars[1:min(20, length(vars))])], obs_ix=obs_ix, id_var=glb_id_var, category_var=glbFeatsCategory)

#model_diags_df[row.names(model_diags_df) %in% names(outliers$rstudent)[c(1:2)], ]
#ctgry_diags_df <- model_diags_df[model_diags_df[, glbFeatsCategory] %in% c("Unknown#0"), ]
#myplot_parcoord(obs_df=ctgry_diags_df[, c(glb_id_var, glbFeatsCategory, ".rstudent", ".dffits", ".hatvalues", glb_rsp_var, "startprice.log10.predict.RFE.X.glmnet", indep_vars[1:20])], obs_ix=row.names(ctgry_diags_df) %in% names(outliers$rstudent)[1], id_var=glb_id_var, category_var=glbFeatsCategory)
#table(glbObsFit[model_diags_df[, glbFeatsCategory] %in% c("iPad1#1"), "startprice.log10.cut.fctr"])
#glbObsFit[model_diags_df[, glbFeatsCategory] %in% c("iPad1#1"), c(glb_id_var, "startprice")]

# No outliers & .dffits == NaN
#myplot_parcoord(obs_df=model_diags_df[, c(glb_id_var, glbFeatsCategory, glb_rsp_var, "startprice.log10.predict.RFE.X.glmnet", indep_vars[1:10])], obs_ix=seq(1:nrow(model_diags_df))[is.na(model_diags_df$.dffits)], id_var=glb_id_var, category_var=glbFeatsCategory)

# Modify mdlId to (build & extract) "<FamilyId>#<Fit|Trn>#<caretMethod>#<preProc1.preProc2>#<samplingMethod>"
glb_models_lst <- list(); glb_models_df <- data.frame()
# Regression
if (glb_is_regression) {
    glbMdlMethods <- c(NULL
        # deterministic
            #, "lm", # same as glm
            , "glm", "bayesglm", "glmnet"
            , "rpart"
        # non-deterministic
            , "gbm", "rf" 
        # Unknown
            , "nnet" , "avNNet" # runs 25 models per cv sample for tunelength=5
            , "svmLinear", "svmLinear2"
            , "svmPoly" # runs 75 models per cv sample for tunelength=5
            , "svmRadial" 
            , "earth"
            , "bagEarth" # Takes a long time
        )
} else
# Classification - Add ada (auto feature selection)
    if (glb_is_binomial)
        glbMdlMethods <- c(NULL
        # deterministic                     
            , "bagEarth" # Takes a long time        
            , "glm", "bayesglm", "glmnet"
            , "nnet"
            , "rpart"
        # non-deterministic        
            , "gbm"
            , "avNNet" # runs 25 models per cv sample for tunelength=5      
            , "rf"
        # Unknown
            , "lda", "lda2"
                # svm models crash when predict is called -> internal to kernlab it should call predict without .outcome
            , "svmLinear", "svmLinear2"
            , "svmPoly" # runs 75 models per cv sample for tunelength=5
            , "svmRadial" 
            , "earth"
        ) else
        glbMdlMethods <- c(NULL
        # deterministic
            ,"glmnet"
        # non-deterministic 
            ,"rf"       
        # Unknown
            ,"gbm","rpart"
        )

glbMdlFamilies <- list(); glb_mdl_feats_lst <- list()
# family: Choose from c("RFE.X", "CSM.X", "All.X", "Best.Interact")
#   methods: Choose from c(NULL, <method>, glbMdlMethods) 
#glbMdlFamilies[["RFE.X"]] <- c("glmnet", "glm") # non-NULL vector is mandatory
glbMdlFamilies[["All.X"]] <- c("glmnet","rpart","rf")  # non-NULL vector is mandatory
#glbMdlFamilies[["Best.Interact"]] <- "glmnet" # non-NULL vector is mandatory

# Check if interaction features make RFE better
# glbMdlFamilies[["CSM.X"]] <- setdiff(glbMdlMethods, c("lda", "lda2")) # crashing due to category:.clusterid ??? #c("glmnet", "glm") # non-NULL list is mandatory
# glb_mdl_feats_lst[["CSM.X"]] <- c(NULL
#     , <comma-separated-features-vector>
#                                   )
# dAFeats.CSM.X %<d-% c(NULL
#     # Interaction feats up to varImp(RFE.X.glmnet) >= 50
#     , <comma-separated-features-vector>
#     , setdiff(myextract_actual_feats(predictors(rfe_fit_results)), c(NULL
#                , <comma-separated-features-vector>
#                                                                       ))    
#                                   )
# glb_mdl_feats_lst[["CSM.X"]] <- "%<d-% dAFeats.CSM.X"

glbMdlFamilies[["Final"]] <- c(NULL) # NULL vector acceptable

glbMdlAllowParallel <- list()
glbMdlAllowParallel[["Max.cor.Y##rcv#rpart"]] <- FALSE
glbMdlAllowParallel[["Interact.High.cor.Y##rcv#glmnet"]] <- FALSE
glbMdlAllowParallel[["Low.cor.X##rcv#glmnet"]] <- FALSE
glbMdlAllowParallel[["All.X##rcv#glmnet"]] <- FALSE

# Check if tuning parameters make fit better; make it mdlFamily customizable ?
glbMdlTuneParams <- data.frame()
# When glmnet crashes at model$grid with error: ???
glmnetTuneParams <- rbind(data.frame()
                        ,data.frame(parameter = "alpha",  vals = "0.100 0.325 0.550 0.775 1.000")
                        ,data.frame(parameter = "lambda", vals = "9.342e-02")    
                        )
glbMdlTuneParams <- myrbind_df(glbMdlTuneParams,
                               cbind(data.frame(mdlId = "<mdlId>"),
                                     glmnetTuneParams))

    #avNNet    
    #   size=[1] 3 5 7 9; decay=[0] 1e-04 0.001  0.01   0.1; bag=[FALSE]; RMSE=1.3300906 

    #bagEarth
    #   degree=1 [2] 3; nprune=64 128 256 512 [1024]; RMSE=0.6486663 (up)
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "bagEarth", parameter = "nprune", vals = "256")
#     ,data.frame(method = "bagEarth", parameter = "degree", vals = "2")    
# ))

    #earth 
    #   degree=[1]; nprune=2  [9] 17 25 33; RMSE=0.1334478
    
    #gbm 
    #   shrinkage=0.05 [0.10] 0.15 0.20 0.25; n.trees=100 150 200 [250] 300; interaction.depth=[1] 2 3 4 5; n.minobsinnode=[10]; RMSE=0.2008313     
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "gbm", parameter = "shrinkage", min = 0.05, max = 0.25, by = 0.05)
#     ,data.frame(method = "gbm", parameter = "n.trees", min = 100, max = 300, by = 50)
#     ,data.frame(method = "gbm", parameter = "interaction.depth", min = 1, max = 5, by = 1)
#     ,data.frame(method = "gbm", parameter = "n.minobsinnode", min = 10, max = 10, by = 10)
#     #seq(from=0.05,  to=0.25, by=0.05)
# ))

    #glmnet
    #   alpha=0.100 [0.325] 0.550 0.775 1.000; lambda=0.0005232693 0.0024288010 0.0112734954 [0.0523269304] 0.2428800957; RMSE=0.6164891
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "glmnet", parameter = "alpha", vals = "0.550 0.775 0.8875 0.94375 1.000")
#     ,data.frame(method = "glmnet", parameter = "lambda", vals = "9.858855e-05 0.0001971771 0.0009152152 0.0042480525 0.0197177130")    
# ))

    #nnet    
    #   size=3 5 [7] 9 11; decay=0.0001 0.001 0.01 [0.1] 0.2; RMSE=0.9287422
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "nnet", parameter = "size", vals = "3 5 7 9 11")
#     ,data.frame(method = "nnet", parameter = "decay", vals = "0.0001 0.0010 0.0100 0.1000 0.2000")    
# ))

    #rf # Don't bother; results are not deterministic
    #       mtry=2  35  68 [101] 134; RMSE=0.1339974
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "rf", parameter = "mtry", vals = "2 5 9 13 17")
# ))

    #rpart 
    #   cp=0.020 [0.025] 0.030 0.035 0.040; RMSE=0.1770237
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()    
#     ,data.frame(method = "rpart", parameter = "cp", vals = "0.004347826 0.008695652 0.017391304 0.021739130 0.034782609")
# ))
    
    #svmLinear
    #   C=0.01 0.05 [0.10] 0.50 1.00 2.00 3.00 4.00; RMSE=0.1271318; 0.1296718
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "svmLinear", parameter = "C", vals = "0.01 0.05 0.1 0.5 1")
# ))

    #svmLinear2    
    #   cost=0.0625 0.1250 [0.25] 0.50 1.00; RMSE=0.1276354 
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "svmLinear2", parameter = "cost", vals = "0.0625 0.125 0.25 0.5 1")
# ))

    #svmPoly    
    #   degree=[1] 2 3 4 5; scale=0.01 0.05 [0.1] 0.5 1; C=0.50 1.00 [2.00] 3.00 4.00; RMSE=0.1276130
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method="svmPoly", parameter="degree", min=1, max=5, by=1) #seq(1, 5, 1)
#     ,data.frame(method="svmPoly", parameter="scale", vals="0.01, 0.05, 0.1, 0.5, 1")
#     ,data.frame(method="svmPoly", parameter="C", vals="0.50, 1.00, 2.00, 3.00, 4.00")    
# ))

    #svmRadial
    #   sigma=[0.08674323]; C=0.25 0.50 1.00 [2.00] 4.00; RMSE=0.1614957
    
#glb2Sav(); all.equal(sav_models_df, glb_models_df)
    
glb_preproc_methods <- NULL
#     c("YeoJohnson", "center.scale", "range", "pca", "ica", "spatialSign")

# Baseline prediction model feature(s)
glb_Baseline_mdl_var <- NULL # or c("<feat>")

glbMdlMetric_terms <- NULL # or matrix(c(
#                               0,1,2,3,4,
#                               2,0,1,2,3,
#                               4,2,0,1,2,
#                               6,4,2,0,1,
#                               8,6,4,2,0
#                           ), byrow=TRUE, nrow=5)
glbMdlMetricSummary <- NULL # or "<metric_name>"
glbMdlMetricMaximize <- NULL # or FALSE (TRUE is not the default for both classification & regression) 
glbMdlMetricSummaryFn <- NULL # or function(data, lev=NULL, model=NULL) {
#     confusion_mtrx <- t(as.matrix(confusionMatrix(data$pred, data$obs)))
#     #print(confusion_mtrx)
#     #print(confusion_mtrx * glbMdlMetric_terms)
#     metric <- sum(confusion_mtrx * glbMdlMetric_terms) / nrow(data)
#     names(metric) <- glbMdlMetricSummary
#     return(metric)
# }

glbMdlCheckRcv <- FALSE # Turn it on when needed; otherwise takes long time
glb_rcv_n_folds <- 3 # or NULL
glb_rcv_n_repeats <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Model selection criteria
if (glb_is_regression)
    glbMdlMetricsEval <- c("min.RMSE.OOB", "max.R.sq.OOB", "max.Adj.R.sq.fit", "min.RMSE.fit")
    #glbMdlMetricsEval <- c("min.RMSE.fit", "max.R.sq.fit", "max.Adj.R.sq.fit")    
if (glb_is_classification) {
    if (glb_is_binomial)
        glbMdlMetricsEval <- 
            c("max.Accuracy.OOB", "max.AUCROCR.OOB", "max.AUCpROC.OOB", "min.aic.fit", "max.Accuracy.fit") else        
        glbMdlMetricsEval <- c("max.Accuracy.OOB", "max.Kappa.OOB")
}

# select from NULL [no ensemble models], "auto" [all models better than MFO or Baseline], c(mdl_ids in glb_models_lst) [Typically top-rated models in auto]
glb_mdl_ensemble <- NULL
#     "%<d-% setdiff(mygetEnsembleAutoMdlIds(), 'CSM.X.rf')" 
#     c(<comma-separated-mdlIds>
#      )

# Only for classifications; for regressions remove "(.*)\\.prob" form the regex
# tmp_fitobs_df <- glbObsFit[, grep(paste0("^", gsub(".", "\\.", mygetPredictIds$value, fixed = TRUE), "CSM\\.X\\.(.*)\\.prob"), names(glbObsFit), value = TRUE)]; cor_mtrx <- cor(tmp_fitobs_df); cor_vctr <- sort(cor_mtrx[row.names(orderBy(~-Overall, varImp(glb_models_lst[["Ensemble.repeatedcv.glmnet"]])$imp))[1], ]); summary(cor_vctr); cor_vctr
#ntv.glm <- glm(reformulate(indep_vars, glb_rsp_var), family = "binomial", data = glbObsFit)
#step.glm <- step(ntv.glm)

#select from c(NULL, "All.X##rcv#glmnet", "RFE.X##rcv#glmnet", <mdlId>)
# glb_sel_mdl_id <- "All.X##rcv#glmnet" 
glb_sel_mdl_id <- "All.X###glmnet" 

glb_fin_mdl_id <- NULL #select from c(NULL, glb_sel_mdl_id)

glb_dsp_cols <- c(glb_id_var, glbFeatsCategory, glb_rsp_var
#               List critical cols excl. glb_id_var, glbFeatsCategory & glb_rsp_var
                  )

# Output specs
glbOutDataVizFname <- NULL # choose from c(NULL, "<projectId>_obsall.csv")
glb_out_obs <- NULL # select from c(NULL, "all", "new", "trn")
glb_out_vars_lst <- list()
# glb_id_var will be the first output column, by default

if (glb_is_classification && glb_is_binomial) {
    glb_out_vars_lst[["Probability1"]] <- 
        "%<d-% mygetPredictIds(glb_rsp_var, glb_fin_mdl_id)$prob" 
} else {
    glb_out_vars_lst[[glb_rsp_var]] <- 
        "%<d-% mygetPredictIds(glb_rsp_var, glb_fin_mdl_id)$value"
}    
# glb_out_vars_lst[[glb_rsp_var_raw]] <- glb_rsp_var_raw
# glb_out_vars_lst[[paste0(head(unlist(strsplit(mygetPredictIds$value, "")), -1), collapse = "")]] <-

glbOutStackFnames <- NULL #: default
    # c("ebayipads_txt_assoc1_out_bid1_stack.csv") # manual stack
    # c("ebayipads_finmdl_bid1_out_nnet_1.csv") # universal stack
glb_out_pfx <- "seaflow_base_"
glb_save_envir <- FALSE # or TRUE

# Depict process
glb_analytics_pn <- petrinet(name = "glb_analytics_pn",
                        trans_df = data.frame(id = 1:6,
    name = c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df = data.frame(
    begin = c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end   = c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](seaflow_base_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_chunks_df <- myadd_chunk(NULL, "import.data")
```

```
##         label step_major step_minor label_minor    bgn end elapsed
## 1 import.data          1          0           0 17.046  NA      NA
```

## Step `1.0: import data`
#### chunk option: eval=<r condition>

```
## [1] "Reading file ./data/seaflow_21min.csv..."
## [1] "dimensions of data in ./data/seaflow_21min.csv: 72,343 rows x 12 cols"
##   file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big    pe
## 1     203   12       1 25344 27968     34677    14944   32400  2216
## 2     203   12       4 12960 22144     37275    20440   32400  1795
## 3     203   12       6 21424 23008     31725    11253   32384  1901
## 4     203   12       9  7712 14528     28744    10219   32416  1248
## 5     203   12      11 30368 21440     28861     6101   32400 12989
## 6     203   12      15 30032 22704     31221    13488   32400  1883
##   chl_small chl_big     pop
## 1     28237    5072    pico
## 2     36755   14224   ultra
## 3     26640       0    pico
## 4     35392   10704   ultra
## 5     23421    5920 synecho
## 6     27323    6560    pico
##       file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big   pe
## 2499      203   33    6838 22448 27280     33005    13925   32400 1912
## 11023     204  108    1783 19552 26208     37139    19979   32400 1232
## 36839     206  368   26324 30080 27472     36912     1229   32400 1456
## 52644     208  490   10024 27248 20640     31379    16435   32400 5515
## 53218     208  495   11574 20352 11808     32941    15891   32400 5747
## 71599     209  636   29137  7152  4832     43621    25115   32400 1536
##       chl_small chl_big     pop
## 2499      26733    1776    pico
## 11023     35784   13920   ultra
## 36839     23603    8272    pico
## 52644     18637       0 synecho
## 53218     13413       0 synecho
## 71599     35125   10480   ultra
##       file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big    pe
## 72338     209  643   31304 11984  3280     26131     6787   32400  5157
## 72339     209  643   31305 20608 29168     31563    15667   32400  8333
## 72340     209  643   31307 22128 19600     26848     9317   32400 22011
## 72341     209  643   31314 21024 24800     31059    16299   32400  9928
## 72342     209  643   31315 24672 18912     36008    18301   32400  1443
## 72343     209  643   31316  6592  4704     37096    22192   32416 17995
##       chl_small chl_big     pop
## 72338      8187   11344 synecho
## 72339     18491       0 synecho
## 72340     15384    7536 synecho
## 72341     11496       0 synecho
## 72342     29435    4720    pico
## 72343     21288    5024 synecho
## 'data.frame':	72343 obs. of  12 variables:
##  $ file_id  : int  203 203 203 203 203 203 203 203 203 203 ...
##  $ time     : int  12 12 12 12 12 12 12 12 12 12 ...
##  $ cell_id  : int  1 4 6 9 11 15 17 23 24 26 ...
##  $ d1       : int  25344 12960 21424 7712 30368 30032 28704 17008 4896 31312 ...
##  $ d2       : int  27968 22144 23008 14528 21440 22704 21664 7072 13104 22608 ...
##  $ fsc_small: int  34677 37275 31725 28744 28861 31221 37539 38987 25515 34133 ...
##  $ fsc_perp : int  14944 20440 11253 10219 6101 13488 17944 20315 5995 21829 ...
##  $ fsc_big  : int  32400 32400 32384 32416 32400 32400 32400 32416 32400 32400 ...
##  $ pe       : int  2216 1795 1901 1248 12989 1883 2107 1509 1952 2525 ...
##  $ chl_small: int  28237 36755 26640 35392 23421 27323 34627 36680 29621 28205 ...
##  $ chl_big  : int  5072 14224 0 10704 5920 6560 11072 15072 3040 2336 ...
##  $ pop      : chr  "pico" "ultra" "pico" "ultra" ...
##  - attr(*, "comment")= chr "glbObsTrn"
## NULL
```

```
## Loading required package: caTools
```

```
##    file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big   pe
## 2      203   12       4 12960 22144     37275    20440   32400 1795
## 3      203   12       6 21424 23008     31725    11253   32384 1901
## 7      203   12      17 28704 21664     37539    17944   32400 2107
## 8      203   12      23 17008  7072     38987    20315   32416 1509
## 9      203   12      24  4896 13104     25515     5995   32400 1952
## 10     203   12      26 31312 22608     34133    21829   32400 2525
##    chl_small chl_big   pop
## 2      36755   14224 ultra
## 3      26640       0  pico
## 7      34627   11072 ultra
## 8      36680   15072 ultra
## 9      29621    3040  pico
## 10     28205    2336  pico
##       file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big   pe
## 4588      203   50   12679  7792 10944     42805    25224   32416  323
## 18836     204  182   25222 28704 25536     34253    12984   32400 2157
## 28564     206  286    1031  9504  7968     15685      120   32400 1253
## 34094     206  343   18336 10032  5264     42029    25933   32432 1965
## 59804     208  544   29074 28944 23184     34435    15019   32400 1144
## 61659     209  558    1949 22016 17696     36197    19056   32400 3648
##       chl_small chl_big   pop
## 4588      42725   19632  nano
## 18836     34269   10656 ultra
## 28564     17411    6464  pico
## 34094     39389   11632 ultra
## 59804     33128    9696 ultra
## 61659     24864       0  pico
##       file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big    pe
## 72337     209  643   31303 12912 20448     45987    27648   32416  2277
## 72338     209  643   31304 11984  3280     26131     6787   32400  5157
## 72339     209  643   31305 20608 29168     31563    15667   32400  8333
## 72340     209  643   31307 22128 19600     26848     9317   32400 22011
## 72342     209  643   31315 24672 18912     36008    18301   32400  1443
## 72343     209  643   31316  6592  4704     37096    22192   32416 17995
##       chl_small chl_big     pop
## 72337     45832   19520    nano
## 72338      8187   11344 synecho
## 72339     18491       0 synecho
## 72340     15384    7536 synecho
## 72342     29435    4720    pico
## 72343     21288    5024 synecho
## 'data.frame':	36172 obs. of  12 variables:
##  $ file_id  : int  203 203 203 203 203 203 203 203 203 203 ...
##  $ time     : int  12 12 12 12 12 12 12 12 12 12 ...
##  $ cell_id  : int  4 6 17 23 24 26 30 37 44 49 ...
##  $ d1       : int  12960 21424 28704 17008 4896 31312 28272 35520 17920 21296 ...
##  $ d2       : int  22144 23008 21664 7072 13104 22608 26320 28288 8624 21888 ...
##  $ fsc_small: int  37275 31725 37539 38987 25515 34133 41880 50157 33824 34656 ...
##  $ fsc_perp : int  20440 11253 17944 20315 5995 21829 25091 32600 14192 16120 ...
##  $ fsc_big  : int  32400 32384 32400 32416 32400 32400 32400 32400 32400 32384 ...
##  $ pe       : int  1795 1901 2107 1509 1952 2525 1171 2584 1917 712 ...
##  $ chl_small: int  36755 26640 34627 36680 29621 28205 32467 40597 38344 31936 ...
##  $ chl_big  : int  14224 0 11072 15072 3040 2336 6672 12768 12224 5936 ...
##  $ pop      : chr  "ultra" "pico" "ultra" "ultra" ...
##  - attr(*, "comment")= chr "glbObsNew"
##    file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big    pe
## 1      203   12       1 25344 27968     34677    14944   32400  2216
## 4      203   12       9  7712 14528     28744    10219   32416  1248
## 5      203   12      11 30368 21440     28861     6101   32400 12989
## 6      203   12      15 30032 22704     31221    13488   32400  1883
## 11     203   12      27 17056 23616     27157    11832   32400 25787
## 13     203   12      33 29808 22208     30728    13261   32400  8491
##    chl_small chl_big     pop
## 1      28237    5072    pico
## 4      35392   10704   ultra
## 5      23421    5920 synecho
## 6      27323    6560    pico
## 11     16771       0 synecho
## 13     15504     288 synecho
##       file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big    pe
## 1079      203   21    2830  6208   704     37691    23096   32400  3109
## 25026     205  248   16813  3472  6400     36984    21627   32416 18443
## 29781     206  299    4910 33872 25920     35251    18533   32400  2283
## 39746     207  391    6374 36032 28432     37819    26331   32400  1232
## 47707     207  454   27977  6576   144     40365    23328   32400  2232
## 51757     208  484    7680 26096 23152     35299    18139   32416 10640
##       chl_small chl_big     pop
## 1079      45637   21232    nano
## 25026     27243    3968 synecho
## 29781     29125    5248    pico
## 39746     28773    9232    pico
## 47707     41856   12336   ultra
## 51757     21683       0 synecho
##       file_id time cell_id    d1    d2 fsc_small fsc_perp fsc_big    pe
## 72328     209  642   31271  6976  2960     19379     4160   32384  1632
## 72332     209  643   31287  4512 10944     34195    18851   32400 18059
## 72333     209  643   31296 15600 25200     28432    10896   32400  8528
## 72334     209  643   31298  6320  3824     38035    21715   32400  2435
## 72335     209  643   31299  8000 13488     31915    12365   32384  1539
## 72341     209  643   31314 21024 24800     31059    16299   32400  9928
##       chl_small chl_big     pop
## 72328     28715    6896    pico
## 72332     24384    4880 synecho
## 72333     15112       0 synecho
## 72334     38757   12576   ultra
## 72335     37979   11920   ultra
## 72341     11496       0 synecho
## 'data.frame':	36171 obs. of  12 variables:
##  $ file_id  : int  203 203 203 203 203 203 203 203 203 203 ...
##  $ time     : int  12 12 12 12 12 12 12 12 12 12 ...
##  $ cell_id  : int  1 9 11 15 27 33 36 38 42 47 ...
##  $ d1       : int  25344 7712 30368 30032 17056 29808 14688 25520 20192 23104 ...
##  $ d2       : int  27968 14528 21440 22704 23616 22208 9968 16816 29456 28320 ...
##  $ fsc_small: int  34677 28744 28861 31221 27157 30728 37315 37261 33117 33811 ...
##  $ fsc_perp : int  14944 10219 6101 13488 11832 13261 20896 20907 16811 14360 ...
##  $ fsc_big  : int  32400 32416 32400 32400 32400 32400 32400 32416 32400 32416 ...
##  $ pe       : int  2216 1248 12989 1883 25787 8491 1499 2581 6440 1269 ...
##  $ chl_small: int  28237 35392 23421 27323 16771 15504 37907 35259 15371 27651 ...
##  $ chl_big  : int  5072 10704 5920 6560 0 288 11584 6976 992 7808 ...
##  $ pop      : chr  "pico" "ultra" "synecho" "pico" ...
##  - attr(*, "comment")= chr "glbObsTrn"
```

```
## Warning: using .rownames as identifiers for observations
```

```
## [1] "Partition stats:"
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##        pop  .src    .n
## 1     pico  Test 10430
## 2     pico Train 10430
## 3    ultra  Test 10269
## 4    ultra Train 10268
## 5  synecho  Test  9073
## 6  synecho Train  9073
## 7     nano  Test  6349
## 8     nano Train  6349
## 9   crypto  Test    51
## 10  crypto Train    51
##        pop  .src    .n
## 1     pico  Test 10430
## 2     pico Train 10430
## 3    ultra  Test 10269
## 4    ultra Train 10268
## 5  synecho  Test  9073
## 6  synecho Train  9073
## 7     nano  Test  6349
## 8     nano Train  6349
## 9   crypto  Test    51
## 10  crypto Train    51
```

![](seaflow_base_files/figure-html/import.data-1.png) 

```
##    .src    .n
## 1  Test 36172
## 2 Train 36171
```

```
## Loading required package: lazyeval
## Loading required package: gdata
## gdata: read.xls support for 'XLS' (Excel 97-2004) files ENABLED.
## 
## gdata: read.xls support for 'XLSX' (Excel 2007+) files ENABLED.
## 
## Attaching package: 'gdata'
## 
## The following objects are masked from 'package:dplyr':
## 
##     combine, first, last
## 
## The following object is masked from 'package:stats':
## 
##     nobs
## 
## The following object is masked from 'package:utils':
## 
##     object.size
```

```
## [1] "Found 0 duplicates by all features:"
```

```
## NULL
```

```
##          label step_major step_minor label_minor    bgn    end elapsed
## 1  import.data          1          0           0 17.046 33.546    16.5
## 2 inspect.data          2          0           0 33.546     NA      NA
```

## Step `2.0: inspect data`

```
## Loading required package: reshape2
```

![](seaflow_base_files/figure-html/inspect.data-1.png) 

```
##       pop.crypto pop.nano pop.pico pop.synecho pop.ultra
## Test          51     6349    10430        9073     10269
## Train         51     6349    10430        9073     10268
##        pop.crypto  pop.nano  pop.pico pop.synecho pop.ultra
## Test  0.001409930 0.1755225 0.2883446   0.2508294 0.2838936
## Train 0.001409969 0.1755274 0.2883525   0.2508363 0.2838738
## [1] "numeric data missing in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ 0s in glbObsAll: "
##  cell_id fsc_perp       pe  chl_big 
##        2      729      459    11569 
## [1] "numeric data w/ Infs in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ NaNs in glbObsAll: "
## named integer(0)
## [1] "string data missing in glbObsAll: "
## pop 
##   0
```

```
##       pop pop.fctr    .n
## 1    pico     pico 20860
## 2   ultra    ultra 20537
## 3 synecho  synecho 18146
## 4    nano     nano 12698
## 5  crypto   crypto   102
```

![](seaflow_base_files/figure-html/inspect.data-2.png) 

```
##       pop.fctr.crypto pop.fctr.nano pop.fctr.pico pop.fctr.synecho
## Test               51          6349         10430             9073
## Train              51          6349         10430             9073
##       pop.fctr.ultra
## Test           10269
## Train          10268
##       pop.fctr.crypto pop.fctr.nano pop.fctr.pico pop.fctr.synecho
## Test      0.001409930     0.1755225     0.2883446        0.2508294
## Train     0.001409969     0.1755274     0.2883525        0.2508363
##       pop.fctr.ultra
## Test       0.2838936
## Train      0.2838738
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](seaflow_base_files/figure-html/inspect.data-3.png) ![](seaflow_base_files/figure-html/inspect.data-4.png) 

```
##          label step_major step_minor label_minor    bgn    end elapsed
## 2 inspect.data          2          0           0 33.546 54.875   21.33
## 3   scrub.data          2          1           1 54.876     NA      NA
```

### Step `2.1: scrub data`

```
## [1] "numeric data missing in : "
## named integer(0)
## [1] "numeric data w/ 0s in : "
##  cell_id fsc_perp       pe  chl_big 
##        2      729      459    11569 
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## pop 
##   0
```

```
##            label step_major step_minor label_minor    bgn    end elapsed
## 3     scrub.data          2          1           1 54.876 58.347   3.471
## 4 transform.data          2          2           2 58.347     NA      NA
```

### Step `2.2: transform data`

```
## [1] "Creating new feature: .pos..."
```

```
##              label step_major step_minor label_minor    bgn  end elapsed
## 4   transform.data          2          2           2 58.347 58.4   0.053
## 5 extract.features          3          0           0 58.401   NA      NA
```

## Step `3.0: extract features`

```
##                  label step_major step_minor label_minor   bgn end elapsed
## 1 extract.features_bgn          1          0           0 58.46  NA      NA
```

```
##                                 label step_major step_minor label_minor
## 1                extract.features_bgn          1          0           0
## 2 extract.features_factorize.str.vars          2          0           0
##      bgn    end elapsed
## 1 58.460 58.475   0.015
## 2 58.476     NA      NA
```

```
##    pop   .src 
##  "pop" ".src"
```

```
##                                 label step_major step_minor label_minor
## 2 extract.features_factorize.str.vars          2          0           0
## 3                extract.features_end          3          0           0
##      bgn    end elapsed
## 2 58.476 58.497   0.021
## 3 58.498     NA      NA
```

```
##                                 label step_major step_minor label_minor
## 2 extract.features_factorize.str.vars          2          0           0
## 1                extract.features_bgn          1          0           0
##      bgn    end elapsed duration
## 2 58.476 58.497   0.021    0.021
## 1 58.460 58.475   0.015    0.015
## [1] "Total Elapsed Time: 58.497 secs"
```

![](seaflow_base_files/figure-html/extract.features-1.png) 

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](seaflow_base_files/figure-html/extract.features-2.png) 

```
##                 label step_major step_minor label_minor    bgn   end
## 5    extract.features          3          0           0 58.401 59.91
## 6 manage.missing.data          3          1           1 59.911    NA
##   elapsed
## 5   1.509
## 6      NA
```

### Step `3.1: manage missing data`

```
## [1] "numeric data missing in : "
## named integer(0)
## [1] "numeric data w/ 0s in : "
##  cell_id fsc_perp       pe  chl_big 
##        2      729      459    11569 
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## pop 
##   0
```

```
## [1] "numeric data missing in : "
## named integer(0)
## [1] "numeric data w/ 0s in : "
##  cell_id fsc_perp       pe  chl_big 
##        2      729      459    11569 
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## pop 
##   0
```

```
##                 label step_major step_minor label_minor    bgn    end
## 6 manage.missing.data          3          1           1 59.911 61.203
## 7        cluster.data          3          2           2 61.204     NA
##   elapsed
## 6   1.292
## 7      NA
```

## Step `3.2: cluster data`

```
##                     label step_major step_minor label_minor    bgn    end
## 7            cluster.data          3          2           2 61.204 61.329
## 8 partition.data.training          4          0           0 61.329     NA
##   elapsed
## 7   0.125
## 8      NA
```

## Step `4.0: partition data training`

```
## [1] "Newdata contains non-NA data for pop.fctr; setting OOB to Newdata"
```

```
##     pop.crypto pop.nano pop.pico pop.synecho pop.ultra
## Fit         51     6349    10430        9073     10268
## OOB         51     6349    10430        9073     10269
##      pop.crypto  pop.nano  pop.pico pop.synecho pop.ultra
## Fit 0.001409969 0.1755274 0.2883525   0.2508363 0.2838738
## OOB 0.001409930 0.1755225 0.2883446   0.2508294 0.2838936
##   .category .n.Fit .n.OOB .n.Tst .freqRatio.Fit .freqRatio.OOB
## 1    .dummy  36171  36172  36172              1              1
##   .freqRatio.Tst
## 1              1
```

```
## [1] "glbObsAll: "
```

```
## [1] 72343    19
```

```
## [1] "glbObsTrn: "
```

```
## [1] 36171    19
```

```
## [1] "glbObsFit: "
```

```
## [1] 36171    18
```

```
## [1] "glbObsOOB: "
```

```
## [1] 36172    18
```

```
## [1] "glbObsNew: "
```

```
## [1] 36172    18
```

```
##                     label step_major step_minor label_minor    bgn    end
## 8 partition.data.training          4          0           0 61.329 62.591
## 9         select.features          5          0           0 62.592     NA
##   elapsed
## 8   1.263
## 9      NA
```

## Step `5.0: select features`

```
## Warning in cor(data.matrix(entity_df[, sel_feats]), y =
## as.numeric(entity_df[, : the standard deviation is zero
```

```
##                  id         cor.y exclude.as.feat    cor.y.abs
## chl_big     chl_big -0.2484344524               0 0.2484344524
## chl_small chl_small -0.1960481902               0 0.1960481902
## fsc_perp   fsc_perp -0.1277406894               0 0.1277406894
## fsc_small fsc_small -0.1175349950               0 0.1175349950
## pe               pe  0.1065189318               0 0.1065189318
## d1               d1  0.0626468266               0 0.0626468266
## d2               d2  0.0455025219               0 0.0455025219
## file_id     file_id  0.0433727666               0 0.0433727666
## time           time  0.0427599552               0 0.0427599552
## .rownames .rownames  0.0419772692               1 0.0419772692
## .pos           .pos  0.0419772692               0 0.0419772692
## fsc_big     fsc_big -0.0314776118               0 0.0314776118
## cell_id     cell_id  0.0020373002               0 0.0020373002
## .rnorm       .rnorm -0.0002310637               0 0.0002310637
## .category .category            NA               1           NA
```

```
## [1] "cor(.pos, time)=0.9978"
## [1] "cor(pop.fctr, .pos)=0.0420"
## [1] "cor(pop.fctr, time)=0.0428"
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, obs_df =
## glbObsTrn, : Identified .pos as highly correlated with time
```

```
## [1] "cor(file_id, time)=0.9901"
## [1] "cor(pop.fctr, file_id)=0.0434"
## [1] "cor(pop.fctr, time)=0.0428"
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, obs_df =
## glbObsTrn, : Identified time as highly correlated with file_id
```

```
## [1] "cor(fsc_perp, fsc_small)=0.9506"
## [1] "cor(pop.fctr, fsc_perp)=-0.1277"
## [1] "cor(pop.fctr, fsc_small)=-0.1175"
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, obs_df =
## glbObsTrn, : Identified fsc_small as highly correlated with fsc_perp
```

```
## [1] "cor(chl_big, chl_small)=0.8460"
## [1] "cor(pop.fctr, chl_big)=-0.2484"
## [1] "cor(pop.fctr, chl_small)=-0.1960"
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, obs_df =
## glbObsTrn, : Identified chl_small as highly correlated with chl_big
```

```
## [1] "cor(d1, d2)=0.8136"
## [1] "cor(pop.fctr, d1)=0.0626"
## [1] "cor(pop.fctr, d2)=0.0455"
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, obs_df =
## glbObsTrn, : Identified d2 as highly correlated with d1
```

```
##                  id         cor.y exclude.as.feat    cor.y.abs cor.high.X
## pe               pe  0.1065189318               0 0.1065189318       <NA>
## d1               d1  0.0626468266               0 0.0626468266       <NA>
## d2               d2  0.0455025219               0 0.0455025219         d1
## file_id     file_id  0.0433727666               0 0.0433727666       <NA>
## time           time  0.0427599552               0 0.0427599552    file_id
## .pos           .pos  0.0419772692               0 0.0419772692       time
## .rownames .rownames  0.0419772692               1 0.0419772692       <NA>
## cell_id     cell_id  0.0020373002               0 0.0020373002       <NA>
## .rnorm       .rnorm -0.0002310637               0 0.0002310637       <NA>
## fsc_big     fsc_big -0.0314776118               0 0.0314776118       <NA>
## fsc_small fsc_small -0.1175349950               0 0.1175349950   fsc_perp
## fsc_perp   fsc_perp -0.1277406894               0 0.1277406894       <NA>
## chl_small chl_small -0.1960481902               0 0.1960481902    chl_big
## chl_big     chl_big -0.2484344524               0 0.2484344524       <NA>
## .category .category            NA               1           NA       <NA>
##            freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## pe          4.153846  2.068231e+01   FALSE FALSE            FALSE
## d1          1.079365  6.673855e+00   FALSE FALSE            FALSE
## d2          1.025000  7.143845e+00   FALSE FALSE            FALSE
## file_id     1.018135  1.935252e-02   FALSE FALSE            FALSE
## time        1.040000  1.747256e+00   FALSE FALSE            FALSE
## .pos        1.000000  1.000000e+02   FALSE FALSE            FALSE
## .rownames   1.000000  1.000000e+02   FALSE FALSE            FALSE
## cell_id     1.000000  6.191977e+01   FALSE FALSE            FALSE
## .rnorm      1.000000  9.986453e+01   FALSE FALSE            FALSE
## fsc_big     2.153882  1.658787e-02   FALSE FALSE            FALSE
## fsc_small   1.000000  3.047745e+01   FALSE FALSE            FALSE
## fsc_perp   21.588235  2.924719e+01   FALSE FALSE            FALSE
## chl_small   1.000000  3.781206e+01   FALSE FALSE            FALSE
## chl_big   116.755102  5.197534e+00   FALSE  TRUE            FALSE
## .category   0.000000  2.764646e-03    TRUE  TRUE               NA
```

```
## Warning in myplot_scatter(plt_feats_df, "percentUnique", "freqRatio",
## colorcol_name = "nzv", : converting nzv to class:factor
```

```
## Warning: Removed 7 rows containing missing values (geom_point).
```

```
## Warning: Removed 7 rows containing missing values (geom_point).
```

```
## Warning: Removed 7 rows containing missing values (geom_point).
```

![](seaflow_base_files/figure-html/select.features-1.png) 

```
##                  id      cor.y exclude.as.feat cor.y.abs cor.high.X
## chl_big     chl_big -0.2484345               0 0.2484345       <NA>
## .category .category         NA               1        NA       <NA>
##           freqRatio percentUnique zeroVar  nzv is.cor.y.abs.low
## chl_big    116.7551   5.197533936   FALSE TRUE            FALSE
## .category    0.0000   0.002764646    TRUE TRUE               NA
```

![](seaflow_base_files/figure-html/select.features-2.png) 

```
## [1] "numeric data missing in : "
## named integer(0)
## [1] "numeric data w/ 0s in : "
##  cell_id fsc_perp       pe  chl_big 
##        2      729      459    11569 
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
##  pop .lcn 
##    0    0
```

```
## [1] "glb_feats_df:"
```

```
## [1] 15 12
```

```
##                id exclude.as.feat rsp_var
## pop.fctr pop.fctr            TRUE    TRUE
```

```
##                id cor.y exclude.as.feat cor.y.abs cor.high.X freqRatio
## pop.fctr pop.fctr    NA            TRUE        NA       <NA>        NA
##          percentUnique zeroVar nzv is.cor.y.abs.low interaction.feat
## pop.fctr            NA      NA  NA               NA               NA
##          shapiro.test.p.value rsp_var_raw rsp_var
## pop.fctr                   NA          NA    TRUE
```

```
## [1] "glb_feats_df vs. glbObsAll: "
```

```
## character(0)
```

```
## [1] "glbObsAll vs. glb_feats_df: "
```

```
## character(0)
```

```
##              label step_major step_minor label_minor    bgn    end elapsed
## 9  select.features          5          0           0 62.592 68.734   6.143
## 10      fit.models          6          0           0 68.735     NA      NA
```

## Step `6.0: fit models`

```r
fit.models_0_chunk_df <- myadd_chunk(NULL, "fit.models_0_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_0_bgn          1          0       setup 69.375  NA      NA
```

```r
# load(paste0(glb_out_pfx, "dsk.RData"))

get_model_sel_frmla <- function() {
    model_evl_terms <- c(NULL)
    # min.aic.fit might not be avl
    lclMdlEvlCriteria <- 
        glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)]
    for (metric in lclMdlEvlCriteria)
        model_evl_terms <- c(model_evl_terms, 
                             ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
    if (glb_is_classification && glb_is_binomial)
        model_evl_terms <- c(model_evl_terms, "-", "opt.prob.threshold.OOB")
    model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse = " "))
    return(model_sel_frmla)
}

get_dsp_models_df <- function() {
    dsp_models_cols <- c("id", 
                    glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                    grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
    dsp_models_df <- 
        #orderBy(get_model_sel_frmla(), glb_models_df)[, c("id", glbMdlMetricsEval)]
        orderBy(get_model_sel_frmla(), glb_models_df)[, dsp_models_cols]    
    nCvMdl <- sapply(glb_models_lst, function(mdl) nrow(mdl$results))
    nParams <- sapply(glb_models_lst, function(mdl) ifelse(mdl$method == "custom", 0, 
        nrow(subset(modelLookup(mdl$method), parameter != "parameter"))))
    
#     nCvMdl <- nCvMdl[names(nCvMdl) != "avNNet"]
#     nParams <- nParams[names(nParams) != "avNNet"]    
    
    if (length(cvMdlProblems <- nCvMdl[nCvMdl <= nParams]) > 0) {
        print("Cross Validation issues:")
        warning("Cross Validation issues:")        
        print(cvMdlProblems)
    }
    
    pltMdls <- setdiff(names(nCvMdl), names(cvMdlProblems))
    pltMdls <- setdiff(pltMdls, names(nParams[nParams == 0]))
    
    # length(pltMdls) == 21
    png(paste0(glb_out_pfx, "bestTune.png"), width = 480 * 2, height = 480 * 4)
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(ceiling(length(pltMdls) / 2.0), 2)))
    pltIx <- 1
    for (mdlId in pltMdls) {
        print(ggplot(glb_models_lst[[mdlId]], highBestTune = TRUE) + labs(title = mdlId),   
              vp = viewport(layout.pos.row = ceiling(pltIx / 2.0), 
                            layout.pos.col = ((pltIx - 1) %% 2) + 1))  
        pltIx <- pltIx + 1
    }
    dev.off()

    if (all(row.names(dsp_models_df) != dsp_models_df$id))
        row.names(dsp_models_df) <- dsp_models_df$id
    return(dsp_models_df)
}
#get_dsp_models_df()

if (glb_is_classification && glb_is_binomial && 
        (length(unique(glbObsFit[, glb_rsp_var])) < 2))
    stop("glbObsFit$", glb_rsp_var, ": contains less than 2 unique values: ",
         paste0(unique(glbObsFit[, glb_rsp_var]), collapse=", "))

max_cor_y_x_vars <- orderBy(~ -cor.y.abs, 
        subset(glb_feats_df, (exclude.as.feat == 0) & !nzv & !is.cor.y.abs.low & 
                                is.na(cor.high.X)))[1:2, "id"]
max_cor_y_x_vars <- max_cor_y_x_vars[!is.na(max_cor_y_x_vars)]

if (!is.null(glb_Baseline_mdl_var)) {
    if ((max_cor_y_x_vars[1] != glb_Baseline_mdl_var) & 
        (glb_feats_df[glb_feats_df$id == max_cor_y_x_vars[1], "cor.y.abs"] > 
         glb_feats_df[glb_feats_df$id == glb_Baseline_mdl_var, "cor.y.abs"]))
        stop(max_cor_y_x_vars[1], " has a higher correlation with ", glb_rsp_var, 
             " than the Baseline var: ", glb_Baseline_mdl_var)
}

glb_model_type <- ifelse(glb_is_regression, "regression", "classification")
    
# Model specs
c("id.prefix", "method", "type",
  # trainControl params
  "preProc.method", "cv.n.folds", "cv.n.repeats", "summary.fn",
  # train params
  "metric", "metric.maximize", "tune.df")
```

```
##  [1] "id.prefix"       "method"          "type"           
##  [4] "preProc.method"  "cv.n.folds"      "cv.n.repeats"   
##  [7] "summary.fn"      "metric"          "metric.maximize"
## [10] "tune.df"
```

```r
# Baseline
if (!is.null(glb_Baseline_mdl_var)) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                            paste0("fit.models_0_", "Baseline"), major.inc = FALSE,
                                    label.minor = "mybaseln_classfr")
    ret_lst <- myfit_mdl(mdl_id="Baseline", 
                         model_method="mybaseln_classfr",
                        indep_vars_vctr=glb_Baseline_mdl_var,
                        rsp_var=glb_rsp_var,
                        fit_df=glbObsFit, OOB_df=glbObsOOB)
}    

# Most Frequent Outcome "MFO" model: mean(y) for regression
#   Not using caret's nullModel since model stats not avl
#   Cannot use rpart for multinomial classification since it predicts non-MFO
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                            paste0("fit.models_0_", "MFO"), major.inc = FALSE,
                                    label.minor = "myMFO_classfr")
```

```
##              label step_major step_minor   label_minor    bgn    end
## 1 fit.models_0_bgn          1          0         setup 69.375 69.416
## 2 fit.models_0_MFO          1          1 myMFO_classfr 69.417     NA
##   elapsed
## 1   0.041
## 2      NA
```

```r
ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
    id.prefix = "MFO", type = glb_model_type, trainControl.method = "none",
    train.method = ifelse(glb_is_regression, "lm", "myMFO_classfr"))),
                        indep_vars = ".rnorm", rsp_var = glb_rsp_var,
                        fit_df = glbObsFit, OOB_df = glbObsOOB)
```

```
## [1] "fitting model: MFO###myMFO_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] crypto  nano    pico    synecho ultra  
## Levels: crypto nano pico synecho ultra
## [1] "unique.prob:"
## y
##        pico       ultra     synecho        nano      crypto 
## 0.288352548 0.283873822 0.250836305 0.175527356 0.001409969 
## [1] "MFO.val:"
## [1] "pico"
##             Length Class      Mode     
## unique.vals 5      factor     numeric  
## unique.prob 5      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   5      -none-     character
## [1] "entr MFO.Classifier$predict"
## [1] "exit MFO.Classifier$predict"
##          Prediction
## Reference crypto  nano  pico synecho ultra
##   crypto       0     0    51       0     0
##   nano         0     0  6349       0     0
##   pico         0     0 10430       0     0
##   synecho      0     0  9073       0     0
##   ultra        0     0 10268       0     0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2883525      0.0000000      0.2836875      0.2930516      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.5019886            NaN 
## [1] "entr MFO.Classifier$predict"
## [1] "exit MFO.Classifier$predict"
##          Prediction
## Reference crypto  nano  pico synecho ultra
##   crypto       0     0    51       0     0
##   nano         0     0  6349       0     0
##   pico         0     0 10430       0     0
##   synecho      0     0  9073       0     0
##   ultra        0     0 10269       0     0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2883446      0.0000000      0.2836797      0.2930435      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.5019886            NaN 
##                    id  feats max.nTuningRuns min.elapsedtime.everything
## 1 MFO###myMFO_classfr .rnorm               0                      0.528
##   min.elapsedtime.final max.Accuracy.fit max.AccuracyLower.fit
## 1                 0.008        0.2883525             0.2836875
##   max.AccuracyUpper.fit max.Kappa.fit max.Accuracy.OOB
## 1             0.2930516             0        0.2883446
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.2836797             0.2930435             0
```

```r
if (glb_is_classification) {
    # "random" model - only for classification; 
    #   none needed for regression since it is same as MFO
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                            paste0("fit.models_0_", "Random"), major.inc = FALSE,
                                    label.minor = "myrandom_classfr")

#stop(here"); glb2Sav(); all.equal(glb_models_df, sav_models_df)    
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Random", type = glb_model_type, trainControl.method = "none",
        train.method = "myrandom_classfr")),
                        indep_vars = ".rnorm", rsp_var = glb_rsp_var,
                        fit_df = glbObsFit, OOB_df = glbObsOOB)
}
```

```
##                 label step_major step_minor      label_minor    bgn    end
## 2    fit.models_0_MFO          1          1    myMFO_classfr 69.417 70.716
## 3 fit.models_0_Random          1          2 myrandom_classfr 70.717     NA
##   elapsed
## 2   1.299
## 3      NA
## [1] "fitting model: Random###myrandom_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
##             Length Class      Mode     
## unique.vals 5      factor     numeric  
## unique.prob 5      table      numeric  
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   5      -none-     character
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       0    6   26       8    11
##   nano        12 1182 1813    1595  1747
##   pico        12 1859 3020    2666  2873
##   synecho     18 1544 2639    2307  2565
##   ultra       11 1746 3023    2574  2914
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##    0.260512565    0.004159934    0.255994892    0.265068719    0.288352548 
## AccuracyPValue  McnemarPValue 
##    1.000000000    0.089297408 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       0   13   17      14     7
##   nano        12 1134 1888    1504  1811
##   pico        17 1800 2973    2635  3005
##   synecho      9 1567 2622    2263  2612
##   ultra       16 1840 2972    2586  2855
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##    0.255031516   -0.003318276    0.250545624    0.259556769    0.288344576 
## AccuracyPValue  McnemarPValue 
##    1.000000000    0.568927472 
##                          id  feats max.nTuningRuns
## 1 Random###myrandom_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      0.297                 0.005        0.2605126
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.2559949             0.2650687   0.004159934
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.2550315             0.2505456             0.2595568
##   max.Kappa.OOB
## 1  -0.003318276
```

```r
# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                        paste0("fit.models_0_", "Max.cor.Y.rcv.*X*"), major.inc = FALSE,
                                    label.minor = "glmnet")
```

```
##                            label step_major step_minor      label_minor
## 3            fit.models_0_Random          1          2 myrandom_classfr
## 4 fit.models_0_Max.cor.Y.rcv.*X*          1          3           glmnet
##      bgn   end elapsed
## 3 70.717 71.71   0.993
## 4 71.711    NA      NA
```

```r
ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
    id.prefix="Max.cor.Y.rcv.1X1", type=glb_model_type, trainControl.method="none",
    train.method="glmnet")),
                    indep_vars=max_cor_y_x_vars, rsp_var=glb_rsp_var, 
                    fit_df=glbObsFit, OOB_df=glbObsOOB)
```

```
## [1] "fitting model: Max.cor.Y.rcv.1X1###glmnet"
## [1] "    indep_vars: fsc_perp,pe"
```

```
## Loading required package: glmnet
## Loading required package: Matrix
## Loaded glmnet 2.0-2
```

```
## Fitting alpha = 0.1, lambda = 0.00689 on full training set
```

![](seaflow_base_files/figure-html/fit.models_0-1.png) ![](seaflow_base_files/figure-html/fit.models_0-2.png) ![](seaflow_base_files/figure-html/fit.models_0-3.png) ![](seaflow_base_files/figure-html/fit.models_0-4.png) ![](seaflow_base_files/figure-html/fit.models_0-5.png) 

```
##             Length Class      Mode     
## a0          500    -none-     numeric  
## beta          5    -none-     list     
## dfmat       500    -none-     numeric  
## df          100    -none-     numeric  
## dim           2    -none-     numeric  
## lambda      100    -none-     numeric  
## dev.ratio   100    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## classnames    5    -none-     character
## grouped       1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     5    -none-     character
## [1] "min lambda > lambdaOpt:"
## [1] "class: crypto:"
##                    fsc_perp            pe 
## -5.5696060298  0.0000651721  0.0001796080 
## [1] "class: nano:"
##                    fsc_perp            pe 
## -3.238735e+00  2.255967e-04 -9.522363e-05 
## [1] "class: pico:"
##                    fsc_perp            pe 
##  5.1769253201 -0.0001677464 -0.0002415023 
## [1] "class: synecho:"
##                    fsc_perp            pe 
##  1.5549139198 -0.0001550318  0.0003764848 
## [1] "class: ultra:"
##                    fsc_perp            pe 
##  2.076502e+00  1.685472e-05 -2.023521e-04 
## [1] "max lambda < lambdaOpt:"
## [1] "class: crypto:"
##                    fsc_perp            pe 
## -5.776664e+00  7.140094e-05  1.953036e-04 
## [1] "class: nano:"
##                    fsc_perp            pe 
## -3.2550035355  0.0002280614 -0.0001003263 
## [1] "class: pico:"
##                    fsc_perp            pe 
##  5.2805746310 -0.0001707064 -0.0002510893 
## [1] "class: synecho:"
##                    fsc_perp            pe 
##  1.6065262793 -0.0001598335  0.0003835819 
## [1] "class: ultra:"
##                    fsc_perp            pe 
##  2.144567e+00  1.592285e-05 -2.104552e-04 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       0    0    0      51     0
##   nano         0 3367  128      18  2836
##   pico         0  218 7321      17  2874
##   synecho      0   48  340    8606    79
##   ultra        0 1203 2746      44  6275
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.7068923      0.6024508      0.7021710      0.7115802      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       0    0    0      51     0
##   nano         0 3357  116      29  2847
##   pico         0  250 7295      15  2870
##   synecho      0   42  387    8574    70
##   ultra        0 1254 2663      33  6319
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.7062092      0.6016196      0.7014849      0.7109004      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##                           id       feats max.nTuningRuns
## 1 Max.cor.Y.rcv.1X1###glmnet fsc_perp,pe               0
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      6.958                  4.92        0.7068923
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1              0.702171             0.7115802     0.6024508
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.7062092             0.7014849             0.7109004
##   max.Kappa.OOB
## 1     0.6016196
```

```r
if (glbMdlCheckRcv) {
    # rcv_n_folds == 1 & rcv_n_repeats > 1 crashes
    for (rcv_n_folds in seq(3, glb_rcv_n_folds + 2, 2))
        for (rcv_n_repeats in seq(1, glb_rcv_n_repeats + 2, 2)) {
            
            # Experiment specific code to avoid caret crash
    #         lcl_tune_models_df <- rbind(data.frame()
    #                             ,data.frame(method = "glmnet", parameter = "alpha", 
    #                                         vals = "0.100 0.325 0.550 0.775 1.000")
    #                             ,data.frame(method = "glmnet", parameter = "lambda",
    #                                         vals = "9.342e-02")    
    #                                     )
            
            ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst =
                list(
                id.prefix = paste0("Max.cor.Y.rcv.", rcv_n_folds, "X", rcv_n_repeats), 
                type = glb_model_type, 
    # tune.df = lcl_tune_models_df,            
                trainControl.method = "repeatedcv",
                trainControl.number = rcv_n_folds, 
                trainControl.repeats = rcv_n_repeats,
                trainControl.classProbs = glb_is_classification,
                trainControl.summaryFunction = glbMdlMetricSummaryFn,
                train.method = "glmnet", train.metric = glbMdlMetricSummary, 
                train.maximize = glbMdlMetricMaximize)),
                                indep_vars = max_cor_y_x_vars, rsp_var = glb_rsp_var, 
                                fit_df = glbObsFit, OOB_df = glbObsOOB)
        }
    # Add parallel coordinates graph of glb_models_df[, glbMdlMetricsEval] to evaluate cv parameters
    tmp_models_cols <- c("id", "max.nTuningRuns",
                        glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                        grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
    print(myplot_parcoord(obs_df = subset(glb_models_df, 
                                          grepl("Max.cor.Y.rcv.", id, fixed = TRUE), 
                                            select = -feats)[, tmp_models_cols],
                          id_var = "id"))
}
        
# Useful for stacking decisions
# fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
#                     paste0("fit.models_0_", "Max.cor.Y[rcv.1X1.cp.0|]"), major.inc = FALSE,
#                                     label.minor = "rpart")
# 
# ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
#     id.prefix = "Max.cor.Y.rcv.1X1.cp.0", type = glb_model_type, trainControl.method = "none",
#     train.method = "rpart",
#     tune.df=data.frame(method="rpart", parameter="cp", min=0.0, max=0.0, by=0.1))),
#                     indep_vars=max_cor_y_x_vars, rsp_var=glb_rsp_var, 
#                     fit_df=glbObsFit, OOB_df=glbObsOOB)

#stop(here"); glb2Sav(); all.equal(glb_models_df, sav_models_df)
# if (glb_is_regression || glb_is_binomial) # For multinomials this model will be run next by default
ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                        id.prefix = "Max.cor.Y", 
                        type = glb_model_type, trainControl.method = "repeatedcv",
                        trainControl.number = glb_rcv_n_folds, 
                        trainControl.repeats = glb_rcv_n_repeats,
                        trainControl.classProbs = glb_is_classification,
                        trainControl.summaryFunction = glbMdlMetricSummaryFn,
                        trainControl.allowParallel = glbMdlAllowParallel,                        
                        train.metric = glbMdlMetricSummary, 
                        train.maximize = glbMdlMetricMaximize,    
                        train.method = "rpart")),
                    indep_vars = max_cor_y_x_vars, rsp_var = glb_rsp_var, 
                    fit_df = glbObsFit, OOB_df = glbObsOOB)
```

```
## [1] "fitting model: Max.cor.Y##rcv#rpart"
## [1] "    indep_vars: fsc_perp,pe"
```

```
## Loading required package: rpart
```

```
## + Fold1.Rep1: cp=0.001515 
## - Fold1.Rep1: cp=0.001515 
## + Fold2.Rep1: cp=0.001515 
## - Fold2.Rep1: cp=0.001515 
## + Fold3.Rep1: cp=0.001515 
## - Fold3.Rep1: cp=0.001515 
## + Fold1.Rep2: cp=0.001515 
## - Fold1.Rep2: cp=0.001515 
## + Fold2.Rep2: cp=0.001515 
## - Fold2.Rep2: cp=0.001515 
## + Fold3.Rep2: cp=0.001515 
## - Fold3.Rep2: cp=0.001515 
## + Fold1.Rep3: cp=0.001515 
## - Fold1.Rep3: cp=0.001515 
## + Fold2.Rep3: cp=0.001515 
## - Fold2.Rep3: cp=0.001515 
## + Fold3.Rep3: cp=0.001515 
## - Fold3.Rep3: cp=0.001515 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.00152 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = "Max.cor.Y", : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

```
## Loading required package: rpart.plot
```

![](seaflow_base_files/figure-html/fit.models_0-6.png) ![](seaflow_base_files/figure-html/fit.models_0-7.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 36171 
## 
##            CP nsplit rel error
## 1 0.350413737      0 1.0000000
## 2 0.165883221      1 0.6495863
## 3 0.069150383      2 0.4837030
## 4 0.014995532      3 0.4145527
## 5 0.001515093      4 0.3995571
## 
## Variable importance
##       pe fsc_perp 
##       65       35 
## 
## Node number 1: 36171 observations,    complexity param=0.3504137
##   predicted class=pico     expected loss=0.7116475  P(node) =1
##     class counts:    51  6349 10430  9073 10268
##    probabilities: 0.001 0.176 0.288 0.251 0.284 
##   left son=2 (26306 obs) right son=3 (9865 obs)
##   Primary splits:
##       pe       < 5006.5  to the left,  improve=8323.884, (0 missing)
##       fsc_perp < 18190.5 to the right, improve=2220.207, (0 missing)
##   Surrogate splits:
##       fsc_perp < 36160   to the left,  agree=0.733, adj=0.02, (0 split)
## 
## Node number 2: 26306 observations,    complexity param=0.1658832
##   predicted class=pico     expected loss=0.6054892  P(node) =0.7272677
##     class counts:     0  5827 10378     1 10100
##    probabilities: 0.000 0.222 0.395 0.000 0.384 
##   left son=4 (14906 obs) right son=5 (11400 obs)
##   Primary splits:
##       fsc_perp < 17190.5 to the right, improve=2784.48400, (0 missing)
##       pe       < 2548    to the right, improve=  92.43883, (0 missing)
##   Surrogate splits:
##       pe < 758.5   to the right, agree=0.57, adj=0.007, (0 split)
## 
## Node number 3: 9865 observations,    complexity param=0.01499553
##   predicted class=synecho  expected loss=0.0803852  P(node) =0.2727323
##     class counts:    51   522    52  9072   168
##    probabilities: 0.005 0.053 0.005 0.920 0.017 
##   left son=6 (621 obs) right son=7 (9244 obs)
##   Primary splits:
##       fsc_perp < 26508   to the right, improve=741.8926, (0 missing)
##       pe       < 6681.5  to the left,  improve=109.4659, (0 missing)
##   Surrogate splits:
##       pe < 37144   to the right, agree=0.941, adj=0.069, (0 split)
## 
## Node number 4: 14906 observations,    complexity param=0.06915038
##   predicted class=ultra    expected loss=0.5458876  P(node) =0.4120981
##     class counts:     0  5638  2499     0  6769
##    probabilities: 0.000 0.378 0.168 0.000 0.454 
##   left son=8 (5332 obs) right son=9 (9574 obs)
##   Primary splits:
##       fsc_perp < 23500   to the right, improve=846.08900, (0 missing)
##       pe       < 2598.5  to the right, improve= 81.31652, (0 missing)
##   Surrogate splits:
##       pe < 3942.5  to the right, agree=0.651, adj=0.023, (0 split)
## 
## Node number 5: 11400 observations
##   predicted class=pico     expected loss=0.3088596  P(node) =0.3151696
##     class counts:     0   189  7879     1  3331
##    probabilities: 0.000 0.017 0.691 0.000 0.292 
## 
## Node number 6: 621 observations
##   predicted class=nano     expected loss=0.257649  P(node) =0.01716845
##     class counts:    44   461     5    75    36
##    probabilities: 0.071 0.742 0.008 0.121 0.058 
## 
## Node number 7: 9244 observations
##   predicted class=synecho  expected loss=0.02672003  P(node) =0.2555638
##     class counts:     7    61    47  8997   132
##    probabilities: 0.001 0.007 0.005 0.973 0.014 
## 
## Node number 8: 5332 observations
##   predicted class=nano     expected loss=0.362153  P(node) =0.1474109
##     class counts:     0  3401   310     0  1621
##    probabilities: 0.000 0.638 0.058 0.000 0.304 
## 
## Node number 9: 9574 observations
##   predicted class=ultra    expected loss=0.4622937  P(node) =0.2646872
##     class counts:     0  2237  2189     0  5148
##    probabilities: 0.000 0.234 0.229 0.000 0.538 
## 
## n= 36171 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 36171 25741 pico (0.0014 0.18 0.29 0.25 0.28)  
##   2) pe< 5006.5 26306 15928 pico (0 0.22 0.39 3.8e-05 0.38)  
##     4) fsc_perp>=17190.5 14906  8137 ultra (0 0.38 0.17 0 0.45)  
##       8) fsc_perp>=23500 5332  1931 nano (0 0.64 0.058 0 0.3) *
##       9) fsc_perp< 23500 9574  4426 ultra (0 0.23 0.23 0 0.54) *
##     5) fsc_perp< 17190.5 11400  3521 pico (0 0.017 0.69 8.8e-05 0.29) *
##   3) pe>=5006.5 9865   793 synecho (0.0052 0.053 0.0053 0.92 0.017)  
##     6) fsc_perp>=26508 621   160 nano (0.071 0.74 0.0081 0.12 0.058) *
##     7) fsc_perp< 26508 9244   247 synecho (0.00076 0.0066 0.0051 0.97 0.014) *
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       0   44    0       7     0
##   nano         0 3862  189      61  2237
##   pico         0  315 7879      47  2189
##   synecho      0   75    1    8997     0
##   ultra        0 1657 3331     132  5148
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.7156562      0.6162647      0.7109764      0.7203014      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       0   43    0       8     0
##   nano         0 3814  188      80  2267
##   pico         0  328 7828      38  2236
##   synecho      0   96    4    8973     0
##   ultra        0 1739 3285      90  5155
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.7124295      0.6119946      0.7077343      0.7170906      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##                     id       feats max.nTuningRuns
## 1 Max.cor.Y##rcv#rpart fsc_perp,pe               5
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                      7.425                 0.319        0.7146147
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.7109764             0.7203014     0.6146897
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1        0.7124295             0.7077343             0.7170906
##   max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1     0.6119946        0.002999769     0.003841578
```

```r
if ((length(glbFeatsDateTime) > 0) && 
    (sum(grepl(paste(names(glbFeatsDateTime), "\\.day\\.minutes\\.poly\\.", sep = ""),
               names(glbObsAll))) > 0)) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Max.cor.Y.Time.Poly"), major.inc = FALSE,
                                    label.minor = "glmnet")

    indepVars <- c(max_cor_y_x_vars, 
            grep(paste(names(glbFeatsDateTime), "\\.day\\.minutes\\.poly\\.", sep = ""),
                        names(glbObsAll), value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
            id.prefix = "Max.cor.Y.Time.Poly", 
            type = glb_model_type, trainControl.method = "repeatedcv",
            trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
            train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
}

if ((length(glbFeatsDateTime) > 0) && 
    (sum(grepl(paste(names(glbFeatsDateTime), "\\.last[[:digit:]]", sep = ""),
               names(glbObsAll))) > 0)) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Max.cor.Y.Time.Lag"), major.inc = FALSE,
                                    label.minor = "glmnet")

    indepVars <- c(max_cor_y_x_vars, 
            grep(paste(names(glbFeatsDateTime), "\\.last[[:digit:]]", sep = ""),
                        names(glbObsAll), value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Time.Lag", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
}

# Interactions.High.cor.Y
if (length(int_feats <- setdiff(setdiff(unique(glb_feats_df$cor.high.X), NA), 
                                subset(glb_feats_df, nzv)$id)) > 0) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Interact.High.cor.Y"), major.inc = FALSE,
                                    label.minor = "glmnet")

    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Interact.High.cor.Y", 
        type = glb_model_type, trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        trainControl.allowParallel = glbMdlAllowParallel,                        
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = c(max_cor_y_x_vars, 
                       paste(max_cor_y_x_vars[1], 
                             setdiff(int_feats, max_cor_y_x_vars[1]), sep = ":")),
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
}    
```

```
##                              label step_major step_minor label_minor
## 4   fit.models_0_Max.cor.Y.rcv.*X*          1          3      glmnet
## 5 fit.models_0_Interact.High.cor.Y          1          4      glmnet
##      bgn    end elapsed
## 4 71.711 90.043  18.332
## 5 90.043     NA      NA
## [1] "fitting model: Interact.High.cor.Y##rcv#glmnet"
## [1] "    indep_vars: fsc_perp,pe,fsc_perp:d1,fsc_perp:file_id,fsc_perp:time"
## + Fold1.Rep1: alpha=0.100, lambda=0.1484 
## - Fold1.Rep1: alpha=0.100, lambda=0.1484 
## + Fold1.Rep1: alpha=0.325, lambda=0.1484 
## - Fold1.Rep1: alpha=0.325, lambda=0.1484 
## + Fold1.Rep1: alpha=0.550, lambda=0.1484 
## - Fold1.Rep1: alpha=0.550, lambda=0.1484 
## + Fold1.Rep1: alpha=0.775, lambda=0.1484 
## - Fold1.Rep1: alpha=0.775, lambda=0.1484 
## + Fold1.Rep1: alpha=1.000, lambda=0.1484 
## - Fold1.Rep1: alpha=1.000, lambda=0.1484 
## + Fold2.Rep1: alpha=0.100, lambda=0.1484 
## - Fold2.Rep1: alpha=0.100, lambda=0.1484 
## + Fold2.Rep1: alpha=0.325, lambda=0.1484 
## - Fold2.Rep1: alpha=0.325, lambda=0.1484 
## + Fold2.Rep1: alpha=0.550, lambda=0.1484 
## - Fold2.Rep1: alpha=0.550, lambda=0.1484 
## + Fold2.Rep1: alpha=0.775, lambda=0.1484 
## - Fold2.Rep1: alpha=0.775, lambda=0.1484 
## + Fold2.Rep1: alpha=1.000, lambda=0.1484 
## - Fold2.Rep1: alpha=1.000, lambda=0.1484 
## + Fold3.Rep1: alpha=0.100, lambda=0.1484 
## - Fold3.Rep1: alpha=0.100, lambda=0.1484 
## + Fold3.Rep1: alpha=0.325, lambda=0.1484 
## - Fold3.Rep1: alpha=0.325, lambda=0.1484 
## + Fold3.Rep1: alpha=0.550, lambda=0.1484 
## - Fold3.Rep1: alpha=0.550, lambda=0.1484 
## + Fold3.Rep1: alpha=0.775, lambda=0.1484 
## - Fold3.Rep1: alpha=0.775, lambda=0.1484 
## + Fold3.Rep1: alpha=1.000, lambda=0.1484 
## - Fold3.Rep1: alpha=1.000, lambda=0.1484 
## + Fold1.Rep2: alpha=0.100, lambda=0.1484 
## - Fold1.Rep2: alpha=0.100, lambda=0.1484 
## + Fold1.Rep2: alpha=0.325, lambda=0.1484 
## - Fold1.Rep2: alpha=0.325, lambda=0.1484 
## + Fold1.Rep2: alpha=0.550, lambda=0.1484 
## - Fold1.Rep2: alpha=0.550, lambda=0.1484 
## + Fold1.Rep2: alpha=0.775, lambda=0.1484 
## - Fold1.Rep2: alpha=0.775, lambda=0.1484 
## + Fold1.Rep2: alpha=1.000, lambda=0.1484 
## - Fold1.Rep2: alpha=1.000, lambda=0.1484 
## + Fold2.Rep2: alpha=0.100, lambda=0.1484 
## - Fold2.Rep2: alpha=0.100, lambda=0.1484 
## + Fold2.Rep2: alpha=0.325, lambda=0.1484 
## - Fold2.Rep2: alpha=0.325, lambda=0.1484 
## + Fold2.Rep2: alpha=0.550, lambda=0.1484 
## - Fold2.Rep2: alpha=0.550, lambda=0.1484 
## + Fold2.Rep2: alpha=0.775, lambda=0.1484 
## - Fold2.Rep2: alpha=0.775, lambda=0.1484 
## + Fold2.Rep2: alpha=1.000, lambda=0.1484 
## - Fold2.Rep2: alpha=1.000, lambda=0.1484 
## + Fold3.Rep2: alpha=0.100, lambda=0.1484 
## - Fold3.Rep2: alpha=0.100, lambda=0.1484 
## + Fold3.Rep2: alpha=0.325, lambda=0.1484 
## - Fold3.Rep2: alpha=0.325, lambda=0.1484 
## + Fold3.Rep2: alpha=0.550, lambda=0.1484 
## - Fold3.Rep2: alpha=0.550, lambda=0.1484 
## + Fold3.Rep2: alpha=0.775, lambda=0.1484 
## - Fold3.Rep2: alpha=0.775, lambda=0.1484 
## + Fold3.Rep2: alpha=1.000, lambda=0.1484 
## - Fold3.Rep2: alpha=1.000, lambda=0.1484 
## + Fold1.Rep3: alpha=0.100, lambda=0.1484 
## - Fold1.Rep3: alpha=0.100, lambda=0.1484 
## + Fold1.Rep3: alpha=0.325, lambda=0.1484 
## - Fold1.Rep3: alpha=0.325, lambda=0.1484 
## + Fold1.Rep3: alpha=0.550, lambda=0.1484 
## - Fold1.Rep3: alpha=0.550, lambda=0.1484 
## + Fold1.Rep3: alpha=0.775, lambda=0.1484 
## - Fold1.Rep3: alpha=0.775, lambda=0.1484 
## + Fold1.Rep3: alpha=1.000, lambda=0.1484
```

```
## Warning: from glmnet Fortran code (error code -40); Convergence for 40th
## lambda value not reached after maxit=100000 iterations; solutions for
## larger lambdas returned
```

```
## - Fold1.Rep3: alpha=1.000, lambda=0.1484 
## + Fold2.Rep3: alpha=0.100, lambda=0.1484 
## - Fold2.Rep3: alpha=0.100, lambda=0.1484 
## + Fold2.Rep3: alpha=0.325, lambda=0.1484 
## - Fold2.Rep3: alpha=0.325, lambda=0.1484 
## + Fold2.Rep3: alpha=0.550, lambda=0.1484 
## - Fold2.Rep3: alpha=0.550, lambda=0.1484 
## + Fold2.Rep3: alpha=0.775, lambda=0.1484 
## - Fold2.Rep3: alpha=0.775, lambda=0.1484 
## + Fold2.Rep3: alpha=1.000, lambda=0.1484 
## - Fold2.Rep3: alpha=1.000, lambda=0.1484 
## + Fold3.Rep3: alpha=0.100, lambda=0.1484 
## - Fold3.Rep3: alpha=0.100, lambda=0.1484 
## + Fold3.Rep3: alpha=0.325, lambda=0.1484 
## - Fold3.Rep3: alpha=0.325, lambda=0.1484 
## + Fold3.Rep3: alpha=0.550, lambda=0.1484 
## - Fold3.Rep3: alpha=0.550, lambda=0.1484 
## + Fold3.Rep3: alpha=0.775, lambda=0.1484 
## - Fold3.Rep3: alpha=0.775, lambda=0.1484 
## + Fold3.Rep3: alpha=1.000, lambda=0.1484 
## - Fold3.Rep3: alpha=1.000, lambda=0.1484 
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.55, lambda = 0.00032 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = "Interact.High.cor.Y", : model's bestTune found at an
## extreme of tuneGrid for parameter: lambda
```

![](seaflow_base_files/figure-html/fit.models_0-8.png) ![](seaflow_base_files/figure-html/fit.models_0-9.png) ![](seaflow_base_files/figure-html/fit.models_0-10.png) ![](seaflow_base_files/figure-html/fit.models_0-11.png) ![](seaflow_base_files/figure-html/fit.models_0-12.png) ![](seaflow_base_files/figure-html/fit.models_0-13.png) 

```
##             Length Class      Mode     
## a0          500    -none-     numeric  
## beta          5    -none-     list     
## dfmat       500    -none-     numeric  
## df          100    -none-     numeric  
## dim           2    -none-     numeric  
## lambda      100    -none-     numeric  
## dev.ratio   100    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## classnames    5    -none-     character
## grouped       1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        5    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     5    -none-     character
## [1] "min lambda > lambdaOpt:"
## [1] "class: crypto:"
##                          pe fsc_perp:time 
## -1.045021e+01  8.394539e-04  1.159727e-08 
## [1] "class: nano:"
##                          fsc_perp               pe      fsc_perp:d1 
##    -3.769735e+00     2.586321e-04    -2.735970e-04    -3.695609e-09 
## fsc_perp:file_id    fsc_perp:time 
##     9.901170e-07    -2.565797e-11 
## [1] "class: pico:"
##                          fsc_perp               pe      fsc_perp:d1 
##     8.111905e+00    -8.876856e-05    -5.868006e-04     4.593000e-09 
## fsc_perp:file_id    fsc_perp:time 
##    -6.311402e-07    -4.548116e-08 
## [1] "class: synecho:"
##                          fsc_perp               pe      fsc_perp:d1 
##     1.823252e+00    -8.566239e-05     7.293630e-04    -7.299486e-10 
## fsc_perp:file_id    fsc_perp:time 
##    -4.709703e-07     6.582778e-08 
## [1] "class: ultra:"
##                          fsc_perp               pe      fsc_perp:d1 
##     4.284787e+00     6.472648e-05    -5.212573e-04     7.352418e-13 
## fsc_perp:file_id 
##     1.241721e-07 
## [1] "max lambda < lambdaOpt:"
## [1] "class: crypto:"
##                    fsc_perp            pe fsc_perp:time 
## -1.051932e+01 -1.899305e-07  8.533002e-04  1.738714e-08 
## [1] "class: nano:"
##                          fsc_perp               pe      fsc_perp:d1 
##    -3.749629e+00     2.642393e-04    -2.799318e-04    -3.700831e-09 
## fsc_perp:file_id    fsc_perp:time 
##     1.012029e-06    -5.241268e-11 
## [1] "class: pico:"
##                          fsc_perp               pe      fsc_perp:d1 
##     8.143659e+00    -8.406236e-05    -5.934844e-04     4.599725e-09 
## fsc_perp:file_id    fsc_perp:time 
##    -6.087946e-07    -4.543786e-08 
## [1] "class: synecho:"
##                          fsc_perp               pe      fsc_perp:d1 
##     1.811857e+00    -8.311089e-05     7.354401e-04    -7.627695e-10 
## fsc_perp:file_id    fsc_perp:time 
##    -4.454775e-07     6.778556e-08 
## [1] "class: ultra:"
##                          fsc_perp               pe fsc_perp:file_id 
##     4.313437e+00     6.982607e-05    -5.281621e-04     1.465863e-07 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto      29    0    0      22     0
##   nano         6 4320   81      26  1916
##   pico         0  124 7890      57  2359
##   synecho      2   40   42    8943    46
##   ultra        0 1487 2809      65  5907
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.7489149      0.6612836      0.7444123      0.7533774      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto      21    0    0      30     0
##   nano        13 4259   82      44  1951
##   pico         0  121 7927      65  2317
##   synecho      3   38   54    8949    29
##   ultra        0 1452 2720      45  6052
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.7521840      0.6655670      0.7477008      0.7566267      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##                                id
## 1 Interact.High.cor.Y##rcv#glmnet
##                                                    feats max.nTuningRuns
## 1 fsc_perp,pe,fsc_perp:d1,fsc_perp:file_id,fsc_perp:time              25
##   min.elapsedtime.everything min.elapsedtime.final max.Accuracy.fit
## 1                    469.183                 11.61        0.7492652
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.7444123             0.7533774     0.6617558
##   max.Accuracy.OOB max.AccuracyLower.OOB max.AccuracyUpper.OOB
## 1         0.752184             0.7477008             0.7566267
##   max.Kappa.OOB max.AccuracySD.fit max.KappaSD.fit
## 1      0.665567        0.003184058     0.004315115
```

```r
# Low.cor.X
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                        paste0("fit.models_0_", "Low.cor.X"), major.inc = FALSE,
                                     label.minor = "glmnet")
```

```
##                              label step_major step_minor label_minor
## 5 fit.models_0_Interact.High.cor.Y          1          4      glmnet
## 6           fit.models_0_Low.cor.X          1          5      glmnet
##       bgn     end elapsed
## 5  90.043 561.782  471.74
## 6 561.783      NA      NA
```

```r
indep_vars <- subset(glb_feats_df, is.na(cor.high.X) & !nzv & 
                              (exclude.as.feat != 1))[, "id"]  
indep_vars <- myadjust_interaction_feats(indep_vars)
# ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
#         id.prefix = "Low.cor.X", 
#         type = glb_model_type, 
#         tune.df = glbMdlTuneParams,        
#         trainControl.method = "repeatedcv",
#         trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
#         trainControl.classProbs = glb_is_classification,
#         trainControl.summaryFunction = glbMdlMetricSummaryFn,
#         trainControl.allowParallel = glbMdlAllowParallel,        
#         train.metric = glbMdlMetricSummary, 
#         train.maximize = glbMdlMetricMaximize,    
#         train.method = "glmnet")),
#         indep_vars = indep_vars, rsp_var = glb_rsp_var, 
#         fit_df = glbObsFit, OOB_df = glbObsOOB)

fit.models_0_chunk_df <- 
    myadd_chunk(fit.models_0_chunk_df, "fit.models_0_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                    label step_major step_minor label_minor     bgn     end
## 6 fit.models_0_Low.cor.X          1          5      glmnet 561.783 561.797
## 7       fit.models_0_end          1          6    teardown 561.798      NA
##   elapsed
## 6   0.015
## 7      NA
```

```r
rm(ret_lst)

glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor     bgn     end elapsed
## 10 fit.models          6          0           0  68.735 561.807 493.072
## 11 fit.models          6          1           1 561.808      NA      NA
```


```r
fit.models_1_chunk_df <- myadd_chunk(NULL, "fit.models_1_bgn", label.minor="setup")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_1_bgn          1          0       setup 563.024  NA      NA
```

```r
#stop(here"); glb2Sav(); all.equal(glb_models_df, sav_models_df)
topindep_var <- NULL; interact_vars <- NULL;
for (mdl_id_pfx in names(glbMdlFamilies)) {
    fit.models_1_chunk_df <- 
        myadd_chunk(fit.models_1_chunk_df, paste0("fit.models_1_", mdl_id_pfx),
                    major.inc = FALSE, label.minor = "setup")

    indep_vars <- NULL;

    if (grepl("\\.Interact", mdl_id_pfx)) {
        if (is.null(topindep_var) && is.null(interact_vars)) {
        #   select best glmnet model upto now
            dsp_models_df <- orderBy(model_sel_frmla <- get_model_sel_frmla(),
                                     glb_models_df)
            dsp_models_df <- subset(dsp_models_df, 
                                    grepl(".glmnet", id, fixed = TRUE))
            bst_mdl_id <- dsp_models_df$id[1]
            mdl_id_pfx <- 
                paste(c(head(unlist(strsplit(bst_mdl_id, "[.]")), -1), "Interact"),
                      collapse=".")
        #   select important features
            if (is.null(bst_featsimp_df <- 
                        myget_feats_importance(glb_models_lst[[bst_mdl_id]]))) {
                warning("Base model for RFE.Interact: ", bst_mdl_id, 
                        " has no important features")
                next
            }    
            
            topindep_ix <- 1
            while (is.null(topindep_var) && (topindep_ix <= nrow(bst_featsimp_df))) {
                topindep_var <- row.names(bst_featsimp_df)[topindep_ix]
                if (grepl(".fctr", topindep_var, fixed=TRUE))
                    topindep_var <- 
                        paste0(unlist(strsplit(topindep_var, ".fctr"))[1], ".fctr")
                if (topindep_var %in% names(glbFeatsInteractionOnly)) {
                    topindep_var <- NULL; topindep_ix <- topindep_ix + 1
                } else break
            }
            
        #   select features with importance > max(10, importance of .rnorm) & is not highest
        #       combine factor dummy features to just the factor feature
            if (length(pos_rnorm <- 
                       grep(".rnorm", row.names(bst_featsimp_df), fixed=TRUE)) > 0)
                imp_rnorm <- bst_featsimp_df[pos_rnorm, 1] else
                imp_rnorm <- NA    
            imp_cutoff <- max(10, imp_rnorm, na.rm=TRUE)
            interact_vars <- 
                tail(row.names(subset(bst_featsimp_df, 
                                      imp > imp_cutoff)), -1)
            if (length(interact_vars) > 0) {
                interact_vars <-
                    myadjust_interaction_feats(myextract_actual_feats(interact_vars))
                interact_vars <- 
                    interact_vars[!grepl(topindep_var, interact_vars, fixed=TRUE)]
            }
            ### bid0_sp only
#             interact_vars <- c(
#     "biddable", "D.ratio.sum.TfIdf.wrds.n", "D.TfIdf.sum.stem.stop.Ratio", "D.sum.TfIdf",
#     "D.TfIdf.sum.post.stop", "D.TfIdf.sum.post.stem", "D.ratio.wrds.stop.n.wrds.n", "D.chrs.uppr.n.log",
#     "D.chrs.n.log", "color.fctr"
#     # , "condition.fctr", "prdl.my.descr.fctr"
#                                 )
#            interact_vars <- setdiff(interact_vars, c("startprice.dgt2.is9", "color.fctr"))
            ###
            indep_vars <- myextract_actual_feats(row.names(bst_featsimp_df))
            indep_vars <- setdiff(indep_vars, topindep_var)
            if (length(interact_vars) > 0) {
                indep_vars <- 
                    setdiff(indep_vars, myextract_actual_feats(interact_vars))
                indep_vars <- c(indep_vars, 
                    paste(topindep_var, setdiff(interact_vars, topindep_var), 
                          sep = "*"))
            } else indep_vars <- union(indep_vars, topindep_var)
        }
    }
    
    if (is.null(indep_vars))
        indep_vars <- glb_mdl_feats_lst[[mdl_id_pfx]]

    if (is.null(indep_vars) && grepl("RFE\\.", mdl_id_pfx))
        indep_vars <- myextract_actual_feats(predictors(rfe_fit_results))
    
    if (is.null(indep_vars))
        indep_vars <- subset(glb_feats_df, !nzv & (exclude.as.feat != 1))[, "id"]
    
    if ((length(indep_vars) == 1) && (grepl("^%<d-%", indep_vars))) {    
        indep_vars <- 
            eval(parse(text = str_trim(unlist(strsplit(indep_vars, "%<d-%"))[2])))
    }    

    indep_vars <- myadjust_interaction_feats(indep_vars)
    
    if (grepl("\\.Interact", mdl_id_pfx)) { 
        # if (method != tail(unlist(strsplit(bst_mdl_id, "[.]")), 1)) next
        if (is.null(glbMdlFamilies[[mdl_id_pfx]])) {
            if (!is.null(glbMdlFamilies[["Best.Interact"]]))
                glbMdlFamilies[[mdl_id_pfx]] <-
                    glbMdlFamilies[["Best.Interact"]]
        }
    }
    
    if (!is.null(glbObsFitOutliers[[mdl_id_pfx]])) {
        fitobs_df <- glbObsFit[!(glbObsFit[, glb_id_var] %in%
                                         glbObsFitOutliers[[mdl_id_pfx]]), ]
    } else fitobs_df <- glbObsFit

    if (is.null(glbMdlFamilies[[mdl_id_pfx]]))
        mdl_methods <- glbMdlMethods else
        mdl_methods <- glbMdlFamilies[[mdl_id_pfx]]    

    for (method in mdl_methods) {
        if (method %in% c("rpart", "rf")) {
            # rpart:    fubar's the tree
            # rf:       skip the scenario w/ .rnorm for speed
            indep_vars <- setdiff(indep_vars, c(".rnorm"))
            #mdl_id <- paste0(mdl_id_pfx, ".no.rnorm")
        } 

        fit.models_1_chunk_df <- myadd_chunk(fit.models_1_chunk_df, 
                            paste0("fit.models_1_", mdl_id_pfx), major.inc = FALSE,
                                    label.minor = method)

        ret_lst <- 
            myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
            id.prefix = mdl_id_pfx, 
            type = glb_model_type, 
            tune.df = glbMdlTuneParams,
            # trainControl.method = "repeatedcv",
            trainControl.method = "none",            
            trainControl.number = glb_rcv_n_folds,
            trainControl.repeats = glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            trainControl.allowParallel = glbMdlAllowParallel,            
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
            train.method = method)),
            indep_vars = indep_vars, rsp_var = glb_rsp_var, 
            fit_df = fitobs_df, OOB_df = glbObsOOB)
        
#         ntv_mdl <- glmnet(x = as.matrix(
#                               fitobs_df[, indep_vars]), 
#                           y = as.factor(as.character(
#                               fitobs_df[, glb_rsp_var])),
#                           family = "multinomial")
#         bgn = 1; end = 100;
#         ntv_mdl <- glmnet(x = as.matrix(
#                               subset(fitobs_df, pop.fctr != "crypto")[bgn:end, indep_vars]), 
#                           y = as.factor(as.character(
#                               subset(fitobs_df, pop.fctr != "crypto")[bgn:end, glb_rsp_var])),
#                           family = "multinomial")
    }
}
```

```
##                label step_major step_minor label_minor     bgn     end
## 1   fit.models_1_bgn          1          0       setup 563.024 563.034
## 2 fit.models_1_All.X          1          1       setup 563.035      NA
##   elapsed
## 1    0.01
## 2      NA
##                label step_major step_minor label_minor     bgn    end
## 2 fit.models_1_All.X          1          1       setup 563.035 563.04
## 3 fit.models_1_All.X          1          2      glmnet 563.041     NA
##   elapsed
## 2   0.006
## 3      NA
## [1] "fitting model: All.X###glmnet"
## [1] "    indep_vars: pe,d1,d2,file_id,time,.pos,cell_id,.rnorm,fsc_big,fsc_small,fsc_perp,chl_small"
## Fitting alpha = 0.1, lambda = 0.00689 on full training set
```

![](seaflow_base_files/figure-html/fit.models_1-1.png) ![](seaflow_base_files/figure-html/fit.models_1-2.png) ![](seaflow_base_files/figure-html/fit.models_1-3.png) ![](seaflow_base_files/figure-html/fit.models_1-4.png) ![](seaflow_base_files/figure-html/fit.models_1-5.png) 

```
##             Length Class      Mode     
## a0          500    -none-     numeric  
## beta          5    -none-     list     
## dfmat       500    -none-     numeric  
## df          100    -none-     numeric  
## dim           2    -none-     numeric  
## lambda      100    -none-     numeric  
## dev.ratio   100    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## classnames    5    -none-     character
## grouped       1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames       12    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     5    -none-     character
## [1] "min lambda > lambdaOpt:"
## [1] "class: crypto:"
##                   chl_small            d1            d2       fsc_big 
## -7.838097e+02  3.077350e-05  8.548977e-06  1.099649e-05  2.758801e-02 
##      fsc_perp     fsc_small            pe 
##  3.595464e-05  2.733441e-05  1.331106e-04 
## [1] "class: nano:"
##                        .pos       cell_id     chl_small            d1 
##  9.895556e+01  1.224234e-07  1.804442e-06  2.464194e-04 -3.137915e-05 
##            d2       file_id      fsc_perp     fsc_small            pe 
## -2.233373e-05  2.450789e-02  7.830669e-05  1.209909e-04 -7.919664e-05 
##          time 
##  1.799684e-04 
## [1] "class: pico:"
##                        .pos        .rnorm     chl_small            d1 
##  3.446880e+02 -1.172227e-06  2.012288e-03 -1.269950e-04  2.379923e-05 
##            d2       file_id       fsc_big      fsc_perp     fsc_small 
##  2.365854e-05 -6.908260e-02 -6.260605e-03 -5.991678e-05 -1.035269e-04 
##            pe          time 
## -2.465363e-04 -5.245340e-04 
## [1] "class: synecho:"
##                   chl_small      fsc_perp     fsc_small            pe 
##  1.252313e+02 -2.206341e-04 -2.672606e-05 -6.397111e-05  3.094823e-04 
## [1] "class: ultra:"
##                   chl_small            d2       file_id       fsc_big 
##  2.149348e+02  5.975987e-05 -7.333515e-07  7.097886e-03 -3.042807e-03 
##      fsc_perp     fsc_small            pe 
## -1.246374e-05  4.196770e-06 -9.984524e-05 
## [1] "max lambda < lambdaOpt:"
## [1] "class: crypto:"
##                   chl_small            d1            d2       fsc_big 
## -8.058186e+02  3.261832e-05  8.727209e-06  1.140573e-05  2.848865e-02 
##      fsc_perp     fsc_small            pe 
##  3.684834e-05  2.813669e-05  1.388502e-04 
## [1] "class: nano:"
##                        .pos       cell_id     chl_small            d1 
##  1.051705e+02  9.500337e-08  1.909630e-06  2.572247e-04 -3.121905e-05 
##            d2       file_id      fsc_perp     fsc_small            pe 
## -2.213118e-05  2.755981e-02  7.788155e-05  1.237508e-04 -8.244714e-05 
##          time 
##  1.971917e-04 
## [1] "class: pico:"
##                        .pos        .rnorm     chl_small            d1 
##  3.483912e+02 -9.441323e-07  2.590738e-03 -1.325554e-04  2.356812e-05 
##            d2       file_id       fsc_big      fsc_perp     fsc_small 
##  2.316151e-05 -7.045651e-02 -6.130446e-03 -5.924043e-05 -1.057377e-04 
##            pe          time 
## -2.546790e-04 -5.296027e-04 
## [1] "class: synecho:"
##                   chl_small      fsc_perp     fsc_small            pe 
##  1.328271e+02 -2.284018e-04 -2.724916e-05 -6.588735e-05  3.178569e-04 
## [1] "class: ultra:"
##                   chl_small            d2       file_id       fsc_big 
##  2.194298e+02  6.043786e-05 -1.385530e-06  1.004731e-02 -2.973058e-03 
##      fsc_perp     fsc_small            pe 
## -1.308556e-05  4.761619e-06 -1.025662e-04 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       8   31    0      12     0
##   nano         0 5302    0       6  1041
##   pico         0    0 8944      68  1418
##   synecho      1    6  152    8905     9
##   ultra        0  947 1469      12  7840
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8570125      0.8072790      0.8533625      0.8606051      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       5   33    0      13     0
##   nano         0 5311    1       7  1030
##   pico         0    1 8977      67  1385
##   synecho      1    6  141    8914    11
##   ultra        0  993 1543       5  7728
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8552195      0.8048991      0.8515509      0.8588309      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##               id
## 1 All.X###glmnet
##                                                                            feats
## 1 pe,d1,d2,file_id,time,.pos,cell_id,.rnorm,fsc_big,fsc_small,fsc_perp,chl_small
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                     21.683                15.083
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.8570125             0.8533625             0.8606051
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1      0.807279        0.8552195             0.8515509
##   max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8588309     0.8048991
##                label step_major step_minor label_minor     bgn     end
## 3 fit.models_1_All.X          1          2      glmnet 563.041 587.399
## 4 fit.models_1_All.X          1          3       rpart 587.400      NA
##   elapsed
## 3  24.358
## 4      NA
## [1] "fitting model: All.X###rpart"
## [1] "    indep_vars: pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small"
## Fitting cp = 0.35 on full training set
```

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 36171 
## 
##          CP nsplit rel error
## 1 0.3504137      0         1
## 
## Node number 1: 36171 observations
##   predicted class=pico  expected loss=0.7116475  P(node) =1
##     class counts:    51  6349 10430  9073 10268
##    probabilities: 0.001 0.176 0.288 0.251 0.284 
## 
## n= 36171 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 36171 25741 pico (0.0014 0.18 0.29 0.25 0.28) *
##          Prediction
## Reference crypto  nano  pico synecho ultra
##   crypto       0     0    51       0     0
##   nano         0     0  6349       0     0
##   pico         0     0 10430       0     0
##   synecho      0     0  9073       0     0
##   ultra        0     0 10268       0     0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2883525      0.0000000      0.2836875      0.2930516      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.5019886            NaN 
##          Prediction
## Reference crypto  nano  pico synecho ultra
##   crypto       0     0    51       0     0
##   nano         0     0  6349       0     0
##   pico         0     0 10430       0     0
##   synecho      0     0  9073       0     0
##   ultra        0     0 10269       0     0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.2883446      0.0000000      0.2836797      0.2930435      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.5019886            NaN 
##              id
## 1 All.X###rpart
##                                                                     feats
## 1 pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                      5.398                 0.583
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1        0.2883525             0.2836875             0.2930516
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1             0        0.2883446             0.2836797
##   max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.2930435             0
##                label step_major step_minor label_minor     bgn     end
## 4 fit.models_1_All.X          1          3       rpart 587.400 594.055
## 5 fit.models_1_All.X          1          4          rf 594.056      NA
##   elapsed
## 4   6.655
## 5      NA
## [1] "fitting model: All.X###rf"
## [1] "    indep_vars: pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small"
```

```
## Loading required package: randomForest
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:gdata':
## 
##     combine
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

![](seaflow_base_files/figure-html/fit.models_1-6.png) 

```
## Fitting mtry = 3 on full training set
```

![](seaflow_base_files/figure-html/fit.models_1-7.png) 

```
##                 Length Class      Mode     
## call                 4 -none-     call     
## type                 1 -none-     character
## predicted        36171 factor     numeric  
## err.rate          3000 -none-     numeric  
## confusion           30 -none-     numeric  
## votes           180855 matrix     numeric  
## oob.times        36171 -none-     numeric  
## classes              5 -none-     character
## importance          11 -none-     numeric  
## importanceSD         0 -none-     NULL     
## localImportance      0 -none-     NULL     
## proximity            0 -none-     NULL     
## ntree                1 -none-     numeric  
## mtry                 1 -none-     numeric  
## forest              14 -none-     list     
## y                36171 factor     numeric  
## test                 0 -none-     NULL     
## inbag                0 -none-     NULL     
## xNames              11 -none-     character
## problemType          1 -none-     character
## tuneValue            1 data.frame list     
## obsLevels            5 -none-     character
##          Prediction
## Reference crypto  nano  pico synecho ultra
##   crypto      51     0     0       0     0
##   nano         0  6349     0       0     0
##   pico         0     0 10430       0     0
##   synecho      0     0     0    9073     0
##   ultra        0     0     0       0 10268
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      1.0000000      1.0000000      0.9998980      1.0000000      0.2883525 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto      47    1    0       3     0
##   nano         4 5812    0       2   531
##   pico         0    0 9774      15   641
##   synecho      0    3    0    9070     0
##   ultra        0  533  612       8  9116
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9349497      0.9123978      0.9323593      0.9374700      0.2883446 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
##           id
## 1 All.X###rf
##                                                                     feats
## 1 pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                     44.948                 44.23
##   max.Accuracy.fit max.AccuracyLower.fit max.AccuracyUpper.fit
## 1                1              0.999898                     1
##   max.Kappa.fit max.Accuracy.OOB max.AccuracyLower.OOB
## 1             1        0.9349497             0.9323593
##   max.AccuracyUpper.OOB max.Kappa.OOB
## 1               0.93747     0.9123978
```

```r
# Check if other preProcess methods improve model performance
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_preProc", major.inc = FALSE,
                label.minor = "preProc")
```

```
##                  label step_major step_minor label_minor     bgn     end
## 5   fit.models_1_All.X          1          4          rf 594.056 646.095
## 6 fit.models_1_preProc          1          5     preProc 646.096      NA
##   elapsed
## 5  52.039
## 6      NA
```

```r
mdl_id <- orderBy(get_model_sel_frmla(), glb_models_df)[1, "id"]
indep_vars_vctr <- trim(unlist(strsplit(glb_models_df[glb_models_df$id == mdl_id,
                                                      "feats"], "[,]")))
method <- tail(unlist(strsplit(mdl_id, "[.]")), 1)
mdl_id_pfx <- paste0(head(unlist(strsplit(mdl_id, "[.]")), -1), collapse = ".")
if (!is.null(glbObsFitOutliers[[mdl_id_pfx]])) {
    fitobs_df <- glbObsFit[!(glbObsFit[, glb_id_var] %in%
                                     glbObsFitOutliers[[mdl_id_pfx]]), ]
} else fitobs_df <- glbObsFit

for (prePr in glb_preproc_methods) {   
    # The operations are applied in this order: 
    #   Box-Cox/Yeo-Johnson transformation, centering, scaling, range, imputation, PCA, ICA then spatial sign.
    
    ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
            id.prefix=mdl_id_pfx, 
            type=glb_model_type, tune.df=glbMdlTuneParams,
            trainControl.method="repeatedcv",
            trainControl.number=glb_rcv_n_folds,
            trainControl.repeats=glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
            train.method=method, train.preProcess=prePr)),
            indep_vars=indep_vars_vctr, rsp_var=glb_rsp_var, 
            fit_df=fitobs_df, OOB_df=glbObsOOB)
}            
    
    # If (All|RFE).X.glm is less accurate than Low.Cor.X.glm
    #   check NA coefficients & filter appropriate terms in indep_vars_vctr
#     if (method == "glm") {
#         orig_glm <- glb_models_lst[[paste0(mdl_id, ".", model_method)]]$finalModel
#         orig_glm <- glb_models_lst[["All.X.glm"]]$finalModel; print(summary(orig_glm))
#         orig_glm <- glb_models_lst[["RFE.X.glm"]]$finalModel; print(summary(orig_glm))
#           require(car)
#           vif_orig_glm <- vif(orig_glm); print(vif_orig_glm)
#           # if vif errors out with "there are aliased coefficients in the model"
#               alias_orig_glm <- alias(orig_glm); alias_complete_orig_glm <- (alias_orig_glm$Complete > 0); alias_complete_orig_glm <- alias_complete_orig_glm[rowSums(alias_complete_orig_glm) > 0, colSums(alias_complete_orig_glm) > 0]; print(alias_complete_orig_glm)
#           print(vif_orig_glm[!is.na(vif_orig_glm) & (vif_orig_glm == Inf)])
#           print(which.max(vif_orig_glm))
#           print(sort(vif_orig_glm[vif_orig_glm >= 1.0e+03], decreasing=TRUE))
#           glbObsFit[c(1143, 3637, 3953, 4105), c("UniqueID", "Popular", "H.P.quandary", "Headline")]
#           glb_feats_df[glb_feats_df$id %in% grep("[HSA]\\.chrs.n.log", glb_feats_df$id, value=TRUE) | glb_feats_df$cor.high.X %in%    grep("[HSA]\\.chrs.n.log", glb_feats_df$id, value=TRUE), ]
#           all.equal(glbObsAll$S.chrs.uppr.n.log, glbObsAll$A.chrs.uppr.n.log)
#           cor(glbObsAll$S.T.herald, glbObsAll$S.T.tribun)
#           mydspObs(Abstract.contains="[Dd]iar", cols=("Abstract"), all=TRUE)
#           subset(glb_feats_df, cor.y.abs <= glb_feats_df[glb_feats_df$id == ".rnorm", "cor.y.abs"])
#         corxx_mtrx <- cor(data.matrix(glbObsAll[, setdiff(names(glbObsAll), myfind_chr_cols_df(glbObsAll))]), use="pairwise.complete.obs"); abs_corxx_mtrx <- abs(corxx_mtrx); diag(abs_corxx_mtrx) <- 0
#           which.max(abs_corxx_mtrx["S.T.tribun", ])
#           abs_corxx_mtrx["A.npnct08.log", "S.npnct08.log"]
#         step_glm <- step(orig_glm)
#     }
    # Since caret does not optimize rpart well
#     if (method == "rpart")
#         ret_lst <- myfit_mdl(mdl_id=paste0(mdl_id_pfx, ".cp.0"), model_method=method,
#                                 indep_vars_vctr=indep_vars_vctr,
#                                 model_type=glb_model_type,
#                                 rsp_var=glb_rsp_var,
#                                 fit_df=glbObsFit, OOB_df=glbObsOOB,        
#             n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))

# User specified
#   Ensure at least 2 vars in each regression; else varImp crashes
# sav_models_lst <- glb_models_lst; sav_models_df <- glb_models_df; sav_featsimp_df <- glb_featsimp_df; all.equal(sav_featsimp_df, glb_featsimp_df)
# glb_models_lst <- sav_models_lst; glb_models_df <- sav_models_df; glm_featsimp_df <- sav_featsimp_df

    # easier to exclude features
# require(gdata) # needed for trim
# mdl_id <- "";
# indep_vars_vctr <- head(subset(glb_models_df, grepl("All\\.X\\.", mdl_id), select=feats)
#                         , 1)[, "feats"]
# indep_vars_vctr <- trim(unlist(strsplit(indep_vars_vctr, "[,]")))
# indep_vars_vctr <- setdiff(indep_vars_vctr, ".rnorm")

    # easier to include features
#stop(here"); sav_models_df <- glb_models_df; glb_models_df <- sav_models_df
# !_sp
# mdl_id <- "csm"; indep_vars_vctr <- c(NULL
#     ,"prdline.my.fctr", "prdline.my.fctr:.clusterid.fctr"
#     ,"prdline.my.fctr*biddable"
#     #,"prdline.my.fctr*startprice.log"
#     #,"prdline.my.fctr*startprice.diff"    
#     ,"prdline.my.fctr*condition.fctr"
#     ,"prdline.my.fctr*D.terms.post.stop.n"
#     #,"prdline.my.fctr*D.terms.post.stem.n"
#     ,"prdline.my.fctr*cellular.fctr"    
# #    ,"<feat1>:<feat2>"
#                                            )
# for (method in glbMdlMethods) {
#     ret_lst <- myfit_mdl(mdl_id=mdl_id, model_method=method,
#                                 indep_vars_vctr=indep_vars_vctr,
#                                 model_type=glb_model_type,
#                                 rsp_var=glb_rsp_var,
#                                 fit_df=glbObsFit, OOB_df=glbObsOOB,
#                     n_cv_folds=glb_rcv_n_folds, tune_models_df=glbMdlTuneParams)
#     csm_mdl_id <- paste0(mdl_id, ".", method)
#     csm_featsimp_df <- myget_feats_importance(glb_models_lst[[paste0(mdl_id, ".",
#                                                                      method)]]);               print(head(csm_featsimp_df))
# }
###

# Ntv.1.lm <- lm(reformulate(indep_vars_vctr, glb_rsp_var), glbObsTrn); print(summary(Ntv.1.lm))

#glb_models_df[, "max.Accuracy.OOB", FALSE]
#varImp(glb_models_lst[["Low.cor.X.glm"]])
#orderBy(~ -Overall, varImp(glb_models_lst[["All.X.2.glm"]])$imp)
#orderBy(~ -Overall, varImp(glb_models_lst[["All.X.3.glm"]])$imp)
#glb_feats_df[grepl("npnct28", glb_feats_df$id), ]

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glbObsFit), 
#                          union(glb_rsp_var, glbFeatsExclude)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]
    
    # template for myfit_mdl
    #   rf is hard-coded in caret to recognize only Accuracy / Kappa evaluation metrics
    #       only for OOB in trainControl ?
    
#     ret_lst <- myfit_mdl_fn(mdl_id=paste0(mdl_id_pfx, ""), model_method=method,
#                             indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var,
#                             fit_df=glbObsFit, OOB_df=glbObsOOB,
#                             n_cv_folds=glb_rcv_n_folds, tune_models_df=glbMdlTuneParams,
#                             model_loss_mtrx=glbMdlMetric_terms,
#                             model_summaryFunction=glbMdlMetricSummaryFn,
#                             model_metric=glbMdlMetricSummary,
#                             model_metric_maximize=glbMdlMetricMaximize)

# Simplify a model
# fit_df <- glbObsFit; glb_mdl <- step(<complex>_mdl)

# Non-caret models
#     rpart_area_mdl <- rpart(reformulate("Area", response=glb_rsp_var), 
#                                data=glbObsFit, #method="class", 
#                                control=rpart.control(cp=0.12),
#                            parms=list(loss=glbMdlMetric_terms))
#     print("rpart_sel_wlm_mdl"); prp(rpart_sel_wlm_mdl)
# 

print(glb_models_df)
```

```
##                                                              id
## MFO###myMFO_classfr                         MFO###myMFO_classfr
## Random###myrandom_classfr             Random###myrandom_classfr
## Max.cor.Y.rcv.1X1###glmnet           Max.cor.Y.rcv.1X1###glmnet
## Max.cor.Y##rcv#rpart                       Max.cor.Y##rcv#rpart
## Interact.High.cor.Y##rcv#glmnet Interact.High.cor.Y##rcv#glmnet
## All.X###glmnet                                   All.X###glmnet
## All.X###rpart                                     All.X###rpart
## All.X###rf                                           All.X###rf
##                                                                                                          feats
## MFO###myMFO_classfr                                                                                     .rnorm
## Random###myrandom_classfr                                                                               .rnorm
## Max.cor.Y.rcv.1X1###glmnet                                                                         fsc_perp,pe
## Max.cor.Y##rcv#rpart                                                                               fsc_perp,pe
## Interact.High.cor.Y##rcv#glmnet                         fsc_perp,pe,fsc_perp:d1,fsc_perp:file_id,fsc_perp:time
## All.X###glmnet                  pe,d1,d2,file_id,time,.pos,cell_id,.rnorm,fsc_big,fsc_small,fsc_perp,chl_small
## All.X###rpart                          pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small
## All.X###rf                             pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small
##                                 max.nTuningRuns min.elapsedtime.everything
## MFO###myMFO_classfr                           0                      0.528
## Random###myrandom_classfr                     0                      0.297
## Max.cor.Y.rcv.1X1###glmnet                    0                      6.958
## Max.cor.Y##rcv#rpart                          5                      7.425
## Interact.High.cor.Y##rcv#glmnet              25                    469.183
## All.X###glmnet                                0                     21.683
## All.X###rpart                                 0                      5.398
## All.X###rf                                    0                     44.948
##                                 min.elapsedtime.final max.Accuracy.fit
## MFO###myMFO_classfr                             0.008        0.2883525
## Random###myrandom_classfr                       0.005        0.2605126
## Max.cor.Y.rcv.1X1###glmnet                      4.920        0.7068923
## Max.cor.Y##rcv#rpart                            0.319        0.7146147
## Interact.High.cor.Y##rcv#glmnet                11.610        0.7492652
## All.X###glmnet                                 15.083        0.8570125
## All.X###rpart                                   0.583        0.2883525
## All.X###rf                                     44.230        1.0000000
##                                 max.AccuracyLower.fit
## MFO###myMFO_classfr                         0.2836875
## Random###myrandom_classfr                   0.2559949
## Max.cor.Y.rcv.1X1###glmnet                  0.7021710
## Max.cor.Y##rcv#rpart                        0.7109764
## Interact.High.cor.Y##rcv#glmnet             0.7444123
## All.X###glmnet                              0.8533625
## All.X###rpart                               0.2836875
## All.X###rf                                  0.9998980
##                                 max.AccuracyUpper.fit max.Kappa.fit
## MFO###myMFO_classfr                         0.2930516   0.000000000
## Random###myrandom_classfr                   0.2650687   0.004159934
## Max.cor.Y.rcv.1X1###glmnet                  0.7115802   0.602450779
## Max.cor.Y##rcv#rpart                        0.7203014   0.614689662
## Interact.High.cor.Y##rcv#glmnet             0.7533774   0.661755789
## All.X###glmnet                              0.8606051   0.807279033
## All.X###rpart                               0.2930516   0.000000000
## All.X###rf                                  1.0000000   1.000000000
##                                 max.Accuracy.OOB max.AccuracyLower.OOB
## MFO###myMFO_classfr                    0.2883446             0.2836797
## Random###myrandom_classfr              0.2550315             0.2505456
## Max.cor.Y.rcv.1X1###glmnet             0.7062092             0.7014849
## Max.cor.Y##rcv#rpart                   0.7124295             0.7077343
## Interact.High.cor.Y##rcv#glmnet        0.7521840             0.7477008
## All.X###glmnet                         0.8552195             0.8515509
## All.X###rpart                          0.2883446             0.2836797
## All.X###rf                             0.9349497             0.9323593
##                                 max.AccuracyUpper.OOB max.Kappa.OOB
## MFO###myMFO_classfr                         0.2930435   0.000000000
## Random###myrandom_classfr                   0.2595568  -0.003318276
## Max.cor.Y.rcv.1X1###glmnet                  0.7109004   0.601619571
## Max.cor.Y##rcv#rpart                        0.7170906   0.611994574
## Interact.High.cor.Y##rcv#glmnet             0.7566267   0.665566956
## All.X###glmnet                              0.8588309   0.804899106
## All.X###rpart                               0.2930435   0.000000000
## All.X###rf                                  0.9374700   0.912397767
##                                 max.AccuracySD.fit max.KappaSD.fit
## MFO###myMFO_classfr                             NA              NA
## Random###myrandom_classfr                       NA              NA
## Max.cor.Y.rcv.1X1###glmnet                      NA              NA
## Max.cor.Y##rcv#rpart                   0.002999769     0.003841578
## Interact.High.cor.Y##rcv#glmnet        0.003184058     0.004315115
## All.X###glmnet                                  NA              NA
## All.X###rpart                                   NA              NA
## All.X###rf                                      NA              NA
```

```r
rm(ret_lst)
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                  label step_major step_minor label_minor     bgn     end
## 6 fit.models_1_preProc          1          5     preProc 646.096 646.144
## 7     fit.models_1_end          1          6    teardown 646.145      NA
##   elapsed
## 6   0.048
## 7      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor     bgn     end elapsed
## 11 fit.models          6          1           1 561.808 646.152  84.344
## 12 fit.models          6          2           2 646.152      NA      NA
```


```r
fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_2_bgn          1          0       setup 647.356  NA      NA
```

```r
plt_models_df <- glb_models_df[, -grep("SD|Upper|Lower", names(glb_models_df))]
for (var in grep("^min.", names(plt_models_df), value=TRUE)) {
    plt_models_df[, sub("min.", "inv.", var)] <- 
        #ifelse(all(is.na(tmp <- plt_models_df[, var])), NA, 1.0 / tmp)
        1.0 / plt_models_df[, var]
    plt_models_df <- plt_models_df[ , -grep(var, names(plt_models_df))]
}
print(plt_models_df)
```

```
##                                                              id
## MFO###myMFO_classfr                         MFO###myMFO_classfr
## Random###myrandom_classfr             Random###myrandom_classfr
## Max.cor.Y.rcv.1X1###glmnet           Max.cor.Y.rcv.1X1###glmnet
## Max.cor.Y##rcv#rpart                       Max.cor.Y##rcv#rpart
## Interact.High.cor.Y##rcv#glmnet Interact.High.cor.Y##rcv#glmnet
## All.X###glmnet                                   All.X###glmnet
## All.X###rpart                                     All.X###rpart
## All.X###rf                                           All.X###rf
##                                                                                                          feats
## MFO###myMFO_classfr                                                                                     .rnorm
## Random###myrandom_classfr                                                                               .rnorm
## Max.cor.Y.rcv.1X1###glmnet                                                                         fsc_perp,pe
## Max.cor.Y##rcv#rpart                                                                               fsc_perp,pe
## Interact.High.cor.Y##rcv#glmnet                         fsc_perp,pe,fsc_perp:d1,fsc_perp:file_id,fsc_perp:time
## All.X###glmnet                  pe,d1,d2,file_id,time,.pos,cell_id,.rnorm,fsc_big,fsc_small,fsc_perp,chl_small
## All.X###rpart                          pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small
## All.X###rf                             pe,d1,d2,file_id,time,.pos,cell_id,fsc_big,fsc_small,fsc_perp,chl_small
##                                 max.nTuningRuns max.Accuracy.fit
## MFO###myMFO_classfr                           0        0.2883525
## Random###myrandom_classfr                     0        0.2605126
## Max.cor.Y.rcv.1X1###glmnet                    0        0.7068923
## Max.cor.Y##rcv#rpart                          5        0.7146147
## Interact.High.cor.Y##rcv#glmnet              25        0.7492652
## All.X###glmnet                                0        0.8570125
## All.X###rpart                                 0        0.2883525
## All.X###rf                                    0        1.0000000
##                                 max.Kappa.fit max.Accuracy.OOB
## MFO###myMFO_classfr               0.000000000        0.2883446
## Random###myrandom_classfr         0.004159934        0.2550315
## Max.cor.Y.rcv.1X1###glmnet        0.602450779        0.7062092
## Max.cor.Y##rcv#rpart              0.614689662        0.7124295
## Interact.High.cor.Y##rcv#glmnet   0.661755789        0.7521840
## All.X###glmnet                    0.807279033        0.8552195
## All.X###rpart                     0.000000000        0.2883446
## All.X###rf                        1.000000000        0.9349497
##                                 max.Kappa.OOB inv.elapsedtime.everything
## MFO###myMFO_classfr               0.000000000                1.893939394
## Random###myrandom_classfr        -0.003318276                3.367003367
## Max.cor.Y.rcv.1X1###glmnet        0.601619571                0.143719460
## Max.cor.Y##rcv#rpart              0.611994574                0.134680135
## Interact.High.cor.Y##rcv#glmnet   0.665566956                0.002131365
## All.X###glmnet                    0.804899106                0.046119079
## All.X###rpart                     0.000000000                0.185253798
## All.X###rf                        0.912397767                0.022247931
##                                 inv.elapsedtime.final
## MFO###myMFO_classfr                      125.00000000
## Random###myrandom_classfr                200.00000000
## Max.cor.Y.rcv.1X1###glmnet                 0.20325203
## Max.cor.Y##rcv#rpart                       3.13479624
## Interact.High.cor.Y##rcv#glmnet            0.08613264
## All.X###glmnet                             0.06629981
## All.X###rpart                              1.71526587
## All.X###rf                                 0.02260909
```

```r
print(myplot_radar(radar_inp_df=plt_models_df))
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have 8.
## Consider specifying shapes manually if you must have them.
```

```
## Warning: Removed 16 rows containing missing values (geom_point).
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have 8.
## Consider specifying shapes manually if you must have them.
```

![](seaflow_base_files/figure-html/fit.models_2-1.png) 

```r
# print(myplot_radar(radar_inp_df=subset(plt_models_df, 
#         !(mdl_id %in% grep("random|MFO", plt_models_df$id, value=TRUE)))))

# Compute CI for <metric>SD
glb_models_df <- mutate(glb_models_df, 
                max.df = ifelse(max.nTuningRuns > 1, max.nTuningRuns - 1, NA),
                min.sd2ci.scaler = ifelse(is.na(max.df), NA, qt(0.975, max.df)))
for (var in grep("SD", names(glb_models_df), value=TRUE)) {
    # Does CI alredy exist ?
    var_components <- unlist(strsplit(var, "SD"))
    varActul <- paste0(var_components[1],          var_components[2])
    varUpper <- paste0(var_components[1], "Upper", var_components[2])
    varLower <- paste0(var_components[1], "Lower", var_components[2])
    if (varUpper %in% names(glb_models_df)) {
        warning(varUpper, " already exists in glb_models_df")
        # Assuming Lower also exists
        next
    }    
    print(sprintf("var:%s", var))
    # CI is dependent on sample size in t distribution; df=n-1
    glb_models_df[, varUpper] <- glb_models_df[, varActul] + 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
    glb_models_df[, varLower] <- glb_models_df[, varActul] - 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
}
```

```
## Warning: max.AccuracyUpper.fit already exists in glb_models_df
```

```
## [1] "var:max.KappaSD.fit"
```

```r
# Plot metrics with CI
plt_models_df <- glb_models_df[, "id", FALSE]
pltCI_models_df <- glb_models_df[, "id", FALSE]
for (var in grep("Upper", names(glb_models_df), value=TRUE)) {
    var_components <- unlist(strsplit(var, "Upper"))
    col_name <- unlist(paste(var_components, collapse=""))
    plt_models_df[, col_name] <- glb_models_df[, col_name]
    for (name in paste0(var_components[1], c("Upper", "Lower"), var_components[2]))
        pltCI_models_df[, name] <- glb_models_df[, name]
}

build_statsCI_data <- function(plt_models_df) {
    mltd_models_df <- melt(plt_models_df, id.vars="id")
    mltd_models_df$data <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) tail(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), "[.]")), 1))
    mltd_models_df$label <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) head(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), 
            paste0(".", mltd_models_df[row_ix, "data"]))), 1))
    #print(mltd_models_df)
    
    return(mltd_models_df)
}
mltd_models_df <- build_statsCI_data(plt_models_df)

mltdCI_models_df <- melt(pltCI_models_df, id.vars="id")
for (row_ix in 1:nrow(mltdCI_models_df)) {
    for (type in c("Upper", "Lower")) {
        if (length(var_components <- unlist(strsplit(
                as.character(mltdCI_models_df[row_ix, "variable"]), type))) > 1) {
            #print(sprintf("row_ix:%d; type:%s; ", row_ix, type))
            mltdCI_models_df[row_ix, "label"] <- var_components[1]
            mltdCI_models_df[row_ix, "data"] <- 
                unlist(strsplit(var_components[2], "[.]"))[2]
            mltdCI_models_df[row_ix, "type"] <- type
            break
        }
    }    
}
wideCI_models_df <- reshape(subset(mltdCI_models_df, select=-variable), 
                            timevar="type", 
        idvar=setdiff(names(mltdCI_models_df), c("type", "value", "variable")), 
                            direction="wide")
#print(wideCI_models_df)
mrgdCI_models_df <- merge(wideCI_models_df, mltd_models_df, all.x=TRUE)
#print(mrgdCI_models_df)

# Merge stats back in if CIs don't exist
goback_vars <- c()
for (var in unique(mltd_models_df$label)) {
    for (type in unique(mltd_models_df$data)) {
        var_type <- paste0(var, ".", type)
        # if this data is already present, next
        if (var_type %in% unique(paste(mltd_models_df$label, mltd_models_df$data,
                                       sep=".")))
            next
        #print(sprintf("var_type:%s", var_type))
        goback_vars <- c(goback_vars, var_type)
    }
}

if (length(goback_vars) > 0) {
    mltd_goback_df <- build_statsCI_data(glb_models_df[, c("id", goback_vars)])
    mltd_models_df <- rbind(mltd_models_df, mltd_goback_df)
}

# mltd_models_df <- merge(mltd_models_df, glb_models_df[, c("id", "model_method")], 
#                         all.x=TRUE)

png(paste0(glb_out_pfx, "models_bar.png"), width=480*3, height=480*2)
#print(gp <- myplot_bar(mltd_models_df, "id", "value", colorcol_name="model_method") + 
print(gp <- myplot_bar(df=mltd_models_df, xcol_name="id", ycol_names="value") + 
        geom_errorbar(data=mrgdCI_models_df, 
            mapping=aes(x=mdl_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
          facet_grid(label ~ data, scales="free") + 
          theme(axis.text.x = element_text(angle = 90,vjust = 0.5)))
```

```
## Warning: Stacking not well defined when ymin != 0
```

```
## Warning: Removed 6 rows containing missing values (geom_errorbar).
```

```r
dev.off()
```

```
## quartz_off_screen 
##                 2
```

```r
print(gp)
```

```
## Warning: Stacking not well defined when ymin != 0
```

```
## Warning: Removed 6 rows containing missing values (geom_errorbar).
```

![](seaflow_base_files/figure-html/fit.models_2-2.png) 

```r
dsp_models_cols <- c("id", 
                    glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                    grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
# if (glb_is_classification && glb_is_binomial) 
#     dsp_models_cols <- c(dsp_models_cols, "opt.prob.threshold.OOB")
print(dsp_models_df <- orderBy(get_model_sel_frmla(), glb_models_df)[, dsp_models_cols])
```

```
##                                id max.Accuracy.OOB max.Kappa.OOB
## 8                      All.X###rf        0.9349497   0.912397767
## 6                  All.X###glmnet        0.8552195   0.804899106
## 5 Interact.High.cor.Y##rcv#glmnet        0.7521840   0.665566956
## 4            Max.cor.Y##rcv#rpart        0.7124295   0.611994574
## 3      Max.cor.Y.rcv.1X1###glmnet        0.7062092   0.601619571
## 1             MFO###myMFO_classfr        0.2883446   0.000000000
## 7                   All.X###rpart        0.2883446   0.000000000
## 2       Random###myrandom_classfr        0.2550315  -0.003318276
```

```r
print(myplot_radar(radar_inp_df = dsp_models_df))
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have 8.
## Consider specifying shapes manually if you must have them.
```

```
## Warning: Removed 6 rows containing missing values (geom_point).
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have 8.
## Consider specifying shapes manually if you must have them.
```

![](seaflow_base_files/figure-html/fit.models_2-3.png) 

```r
print("Metrics used for model selection:"); print(get_model_sel_frmla())
```

```
## [1] "Metrics used for model selection:"
```

```
## ~-max.Accuracy.OOB - max.Kappa.OOB
## <environment: 0x7fbf8f3d8148>
```

```r
print(sprintf("Best model id: %s", dsp_models_df[1, "id"]))
```

```
## [1] "Best model id: All.X###rf"
```

```r
glb_get_predictions <- function(df, mdl_id, rsp_var, prob_threshold_def=NULL, verbose=FALSE) {
    mdl <- glb_models_lst[[mdl_id]]
    
    clmnNames <- mygetPredictIds(rsp_var, mdl_id)
    predct_var_name <- clmnNames$value        
    predct_prob_var_name <- clmnNames$prob
    predct_accurate_var_name <- clmnNames$is.acc
    predct_error_var_name <- clmnNames$err
    predct_erabs_var_name <- clmnNames$err.abs

    if (glb_is_regression) {
        df[, predct_var_name] <- predict(mdl, newdata=df, type="raw")
        if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_var_name) + 
                  facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
                  stat_smooth(method="glm"))

        df[, predct_error_var_name] <- df[, predct_var_name] - df[, glb_rsp_var]
        if (verbose) print(myplot_scatter(df, predct_var_name, predct_error_var_name) + 
                  #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
                  stat_smooth(method="auto"))
        if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_error_var_name) + 
                  #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
                  stat_smooth(method="glm"))
        
        df[, predct_erabs_var_name] <- abs(df[, predct_error_var_name])
        if (verbose) print(head(orderBy(reformulate(c("-", predct_erabs_var_name)), df)))
        
        df[, predct_accurate_var_name] <- (df[, glb_rsp_var] == df[, predct_var_name])
    }

    if (glb_is_classification && glb_is_binomial) {
        prob_threshold <- glb_models_df[glb_models_df$id == mdl_id, 
                                        "opt.prob.threshold.OOB"]
        if (is.null(prob_threshold) || is.na(prob_threshold)) {
            warning("Using default probability threshold: ", prob_threshold_def)
            if (is.null(prob_threshold <- prob_threshold_def))
                stop("Default probability threshold is NULL")
        }
        
        df[, predct_prob_var_name] <- predict(mdl, newdata = df, type = "prob")[, 2]
        df[, predct_var_name] <- 
        		factor(levels(df[, glb_rsp_var])[
    				(df[, predct_prob_var_name] >=
    					prob_threshold) * 1 + 1], levels(df[, glb_rsp_var]))
    
#         if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_var_name) + 
#                   facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
#                   stat_smooth(method="glm"))

        df[, predct_error_var_name] <- df[, predct_var_name] != df[, glb_rsp_var]
#         if (verbose) print(myplot_scatter(df, predct_var_name, predct_error_var_name) + 
#                   #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
#                   stat_smooth(method="auto"))
#         if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_error_var_name) + 
#                   #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
#                   stat_smooth(method="glm"))
        
        # if prediction is a TP (true +ve), measure distance from 1.0
        tp <- which((df[, predct_var_name] == df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[2]))
        df[tp, predct_erabs_var_name] <- abs(1 - df[tp, predct_prob_var_name])
        #rowIx <- which.max(df[tp, predct_erabs_var_name]); df[tp, c(glb_id_var, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a TN (true -ve), measure distance from 0.0
        tn <- which((df[, predct_var_name] == df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[1]))
        df[tn, predct_erabs_var_name] <- abs(0 - df[tn, predct_prob_var_name])
        #rowIx <- which.max(df[tn, predct_erabs_var_name]); df[tn, c(glb_id_var, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a FP (flse +ve), measure distance from 0.0
        fp <- which((df[, predct_var_name] != df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[2]))
        df[fp, predct_erabs_var_name] <- abs(0 - df[fp, predct_prob_var_name])
        #rowIx <- which.max(df[fp, predct_erabs_var_name]); df[fp, c(glb_id_var, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a FN (flse -ve), measure distance from 1.0
        fn <- which((df[, predct_var_name] != df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[1]))
        df[fn, predct_erabs_var_name] <- abs(1 - df[fn, predct_prob_var_name])
        #rowIx <- which.max(df[fn, predct_erabs_var_name]); df[fn, c(glb_id_var, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]

        
        if (verbose) print(head(orderBy(reformulate(c("-", predct_erabs_var_name)), df)))
        
        df[, predct_accurate_var_name] <- (df[, glb_rsp_var] == df[, predct_var_name])
    }    
    
    if (glb_is_classification && !glb_is_binomial) {
        df[, predct_var_name] <- predict(mdl, newdata = df, type = "raw")
        probCls <- predict(mdl, newdata = df, type = "prob")        
        df[, predct_prob_var_name] <- NA
        for (cls in names(probCls)) {
            mask <- (df[, predct_var_name] == cls)
            df[mask, predct_prob_var_name] <- probCls[mask, cls]
        }    
        if (verbose) print(myplot_histogram(df, predct_prob_var_name, 
                                            fill_col_name = predct_var_name))
        if (verbose) print(myplot_histogram(df, predct_prob_var_name, 
                                            facet_frmla = paste0("~", glb_rsp_var)))
        
        df[, predct_error_var_name] <- df[, predct_var_name] != df[, glb_rsp_var]
        
        # if prediction is erroneous, measure predicted class prob from actual class prob
        df[, predct_erabs_var_name] <- 0
        for (cls in names(probCls)) {
            mask <- (df[, glb_rsp_var] == cls) & (df[, predct_error_var_name])
            df[mask, predct_erabs_var_name] <- probCls[mask, cls]
        }    

        df[, predct_accurate_var_name] <- (df[, glb_rsp_var] == df[, predct_var_name])        
    }

    return(df)
}    

#stop(here"); glb2Sav(); glbObsAll <- savObsAll; glbObsTrn <- savObsTrn; glbObsFit <- savObsFit; glbObsOOB <- savObsOOB; sav_models_df <- glb_models_df; glb_models_df <- sav_models_df; glb_featsimp_df <- sav_featsimp_df    

myget_category_stats <- function(obs_df, mdl_id, label) {
    require(dplyr)
    require(lazyeval)
    
    predct_var_name <- mygetPredictIds(glb_rsp_var, mdl_id)$value        
    predct_error_var_name <- mygetPredictIds(glb_rsp_var, mdl_id)$err.abs
    
    if (!predct_var_name %in% names(obs_df))
        obs_df <- glb_get_predictions(obs_df, mdl_id, glb_rsp_var)
    
    tmp_obs_df <- obs_df[, c(glbFeatsCategory, glb_rsp_var, 
                             predct_var_name, predct_error_var_name)]
#     tmp_obs_df <- obs_df %>%
#         dplyr::select_(glbFeatsCategory, glb_rsp_var, predct_var_name, predct_error_var_name) 
    #dplyr::rename(startprice.log10.predict.RFE.X.glmnet.err=error_abs_OOB)
    names(tmp_obs_df)[length(names(tmp_obs_df))] <- paste0("err.abs.", label)
    
    ret_ctgry_df <- tmp_obs_df %>%
        dplyr::group_by_(glbFeatsCategory) %>%
        dplyr::summarise_(#interp(~sum(abs(var)), var=as.name(glb_rsp_var)), 
            interp(~sum(var), var=as.name(paste0("err.abs.", label))), 
            interp(~mean(var), var=as.name(paste0("err.abs.", label))),
            interp(~n()))
    names(ret_ctgry_df) <- c(glbFeatsCategory, 
                             #paste0(glb_rsp_var, ".abs.", label, ".sum"),
                             paste0("err.abs.", label, ".sum"),                             
                             paste0("err.abs.", label, ".mean"), 
                             paste0(".n.", label))
    ret_ctgry_df <- dplyr::ungroup(ret_ctgry_df)
    #colSums(ret_ctgry_df[, -grep(glbFeatsCategory, names(ret_ctgry_df))])
    
    return(ret_ctgry_df)    
}
#print(colSums((ctgry_df <- myget_category_stats(obs_df=glbObsFit, mdl_id="", label="fit"))[, -grep(glbFeatsCategory, names(ctgry_df))]))

if (!is.null(glb_mdl_ensemble)) {
    fit.models_2_chunk_df <- myadd_chunk(fit.models_2_chunk_df, 
                            paste0("fit.models_2_", mdl_id_pfx), major.inc = TRUE, 
                                                label.minor = "ensemble")
    
    mdl_id_pfx <- "Ensemble"

    if (#(glb_is_regression) | 
        ((glb_is_classification) & (!glb_is_binomial)))
        stop("Ensemble models not implemented yet for multinomial classification")
    
    mygetEnsembleAutoMdlIds <- function() {
        tmp_models_df <- orderBy(get_model_sel_frmla(), glb_models_df)
        row.names(tmp_models_df) <- tmp_models_df$id
        mdl_threshold_pos <- 
            min(which(grepl("MFO|Random|Baseline", tmp_models_df$id))) - 1
        mdlIds <- tmp_models_df$id[1:mdl_threshold_pos]
        return(mdlIds[!grepl("Ensemble", mdlIds)])
    }
    
    if (glb_mdl_ensemble == "auto") {
        glb_mdl_ensemble <- mygetEnsembleAutoMdlIds()
        mdl_id_pfx <- paste0(mdl_id_pfx, ".auto")        
    } else if (grepl("^%<d-%", glb_mdl_ensemble)) {
        glb_mdl_ensemble <- eval(parse(text =
                        str_trim(unlist(strsplit(glb_mdl_ensemble, "%<d-%"))[2])))
    }
    
    for (mdl_id in glb_mdl_ensemble) {
        if (!(mdl_id %in% names(glb_models_lst))) {
            warning("Model ", mdl_id, " in glb_model_ensemble not found !")
            next
        }
        glbObsFit <- glb_get_predictions(df = glbObsFit, mdl_id, glb_rsp_var)
        glbObsOOB <- glb_get_predictions(df = glbObsOOB, mdl_id, glb_rsp_var)
    }
    
#mdl_id_pfx <- "Ensemble.RFE"; mdlId <- paste0(mdl_id_pfx, ".glmnet")
#glb_mdl_ensemble <- gsub(mygetPredictIds$value, "", grep("RFE\\.X\\.(?!Interact)", row.names(glb_featsimp_df), perl = TRUE, value = TRUE), fixed = TRUE)
#varImp(glb_models_lst[[mdlId]])
    
#cor_df <- data.frame(cor=cor(glbObsFit[, glb_rsp_var], glbObsFit[, paste(mygetPredictIds$value, glb_mdl_ensemble)], use="pairwise.complete.obs"))
#glbObsFit <- glb_get_predictions(df=glbObsFit, "Ensemble.glmnet", glb_rsp_var);print(colSums((ctgry_df <- myget_category_stats(obs_df=glbObsFit, mdl_id="Ensemble.glmnet", label="fit"))[, -grep(glbFeatsCategory, names(ctgry_df))]))
    
    ### bid0_sp
    #  Better than MFO; models.n=28; min.RMSE.fit=0.0521233; err.abs.fit.sum=7.3631895
    #  old: Top x from auto; models.n= 5; min.RMSE.fit=0.06311047; err.abs.fit.sum=9.5937080
    #  RFE only ;       models.n=16; min.RMSE.fit=0.05148588; err.abs.fit.sum=7.2875091
    #  RFE subset only ;models.n= 5; min.RMSE.fit=0.06040702; err.abs.fit.sum=9.059088
    #  RFE subset only ;models.n= 9; min.RMSE.fit=0.05933167; err.abs.fit.sum=8.7421288
    #  RFE subset only ;models.n=15; min.RMSE.fit=0.0584607; err.abs.fit.sum=8.5902066
    #  RFE subset only ;models.n=17; min.RMSE.fit=0.05496899; err.abs.fit.sum=8.0170431
    #  RFE subset only ;models.n=18; min.RMSE.fit=0.05441577; err.abs.fit.sum=7.837223
    #  RFE subset only ;models.n=16; min.RMSE.fit=0.05441577; err.abs.fit.sum=7.837223
    ### bid0_sp
    ### bid1_sp
    # "auto"; err.abs.fit.sum=76.699774; min.RMSE.fit=0.2186429
    # "RFE.X.*"; err.abs.fit.sum=; min.RMSE.fit=0.221114
    ### bid1_sp

    indep_vars <- paste(mygetPredictIds(glb_rsp_var)$value, glb_mdl_ensemble, sep = "")
    if (glb_is_classification)
        indep_vars <- paste(indep_vars, ".prob", sep = "")
    # Some models in glb_mdl_ensemble might not be fitted e.g. RFE.X.Interact
    indep_vars <- intersect(indep_vars, names(glbObsFit))
    
#     indep_vars <- grep(mygetPredictIds(glb_rsp_var)$value, names(glbObsFit), fixed=TRUE, value=TRUE)
#     if (glb_is_regression)
#         indep_vars <- indep_vars[!grepl("(err\\.abs|accurate)$", indep_vars)]
#     if (glb_is_classification && glb_is_binomial)
#         indep_vars <- grep("prob$", indep_vars, value=TRUE) else
#         indep_vars <- indep_vars[!grepl("err$", indep_vars)]

    #rfe_fit_ens_results <- myrun_rfe(glbObsFit, indep_vars)
    
    for (method in c("glm", "glmnet")) {
        for (trainControlMethod in 
             c("boot", "boot632", "cv", "repeatedcv"
               #, "LOOCV" # tuneLength * nrow(fitDF)
               , "LGOCV", "adaptive_cv"
               #, "adaptive_boot"  #error: adaptive$min should be less than 3 
               #, "adaptive_LGOCV" #error: adaptive$min should be less than 3 
               )) {
            #sav_models_df <- glb_models_df; all.equal(sav_models_df, glb_models_df)
            #glb_models_df <- sav_models_df; print(glb_models_df$id)
                
            if ((method == "glm") && (trainControlMethod != "repeatedcv"))
                # glm used only to identify outliers
                next
            
            ret_lst <- myfit_mdl(
                mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                    id.prefix = paste0(mdl_id_pfx, ".", trainControlMethod), 
                    type = glb_model_type, tune.df = NULL,
                    trainControl.method = trainControlMethod,
                    trainControl.number = glb_rcv_n_folds,
                    trainControl.repeats = glb_rcv_n_repeats,
                    trainControl.classProbs = glb_is_classification,
                    trainControl.summaryFunction = glbMdlMetricSummaryFn,
                    train.metric = glbMdlMetricSummary, 
                    train.maximize = glbMdlMetricMaximize,    
                    train.method = method)),
                indep_vars = indep_vars, rsp_var = glb_rsp_var, 
                fit_df = glbObsFit, OOB_df = glbObsOOB)
        }
    }
    dsp_models_df <- get_dsp_models_df()
}

if (is.null(glb_sel_mdl_id)) 
    glb_sel_mdl_id <- dsp_models_df[1, "id"] else 
    print(sprintf("User specified selection: %s", glb_sel_mdl_id))   
```

```
## [1] "User specified selection: All.X###glmnet"
```

```r
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

![](seaflow_base_files/figure-html/fit.models_2-4.png) ![](seaflow_base_files/figure-html/fit.models_2-5.png) ![](seaflow_base_files/figure-html/fit.models_2-6.png) ![](seaflow_base_files/figure-html/fit.models_2-7.png) ![](seaflow_base_files/figure-html/fit.models_2-8.png) 

```
##             Length Class      Mode     
## a0          500    -none-     numeric  
## beta          5    -none-     list     
## dfmat       500    -none-     numeric  
## df          100    -none-     numeric  
## dim           2    -none-     numeric  
## lambda      100    -none-     numeric  
## dev.ratio   100    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## classnames    5    -none-     character
## grouped       1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames       12    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     5    -none-     character
## [1] "min lambda > lambdaOpt:"
## [1] "class: crypto:"
##                   chl_small            d1            d2       fsc_big 
## -7.838097e+02  3.077350e-05  8.548977e-06  1.099649e-05  2.758801e-02 
##      fsc_perp     fsc_small            pe 
##  3.595464e-05  2.733441e-05  1.331106e-04 
## [1] "class: nano:"
##                        .pos       cell_id     chl_small            d1 
##  9.895556e+01  1.224234e-07  1.804442e-06  2.464194e-04 -3.137915e-05 
##            d2       file_id      fsc_perp     fsc_small            pe 
## -2.233373e-05  2.450789e-02  7.830669e-05  1.209909e-04 -7.919664e-05 
##          time 
##  1.799684e-04 
## [1] "class: pico:"
##                        .pos        .rnorm     chl_small            d1 
##  3.446880e+02 -1.172227e-06  2.012288e-03 -1.269950e-04  2.379923e-05 
##            d2       file_id       fsc_big      fsc_perp     fsc_small 
##  2.365854e-05 -6.908260e-02 -6.260605e-03 -5.991678e-05 -1.035269e-04 
##            pe          time 
## -2.465363e-04 -5.245340e-04 
## [1] "class: synecho:"
##                   chl_small      fsc_perp     fsc_small            pe 
##  1.252313e+02 -2.206341e-04 -2.672606e-05 -6.397111e-05  3.094823e-04 
## [1] "class: ultra:"
##                   chl_small            d2       file_id       fsc_big 
##  2.149348e+02  5.975987e-05 -7.333515e-07  7.097886e-03 -3.042807e-03 
##      fsc_perp     fsc_small            pe 
## -1.246374e-05  4.196770e-06 -9.984524e-05 
## [1] "max lambda < lambdaOpt:"
## [1] "class: crypto:"
##                   chl_small            d1            d2       fsc_big 
## -8.058186e+02  3.261832e-05  8.727209e-06  1.140573e-05  2.848865e-02 
##      fsc_perp     fsc_small            pe 
##  3.684834e-05  2.813669e-05  1.388502e-04 
## [1] "class: nano:"
##                        .pos       cell_id     chl_small            d1 
##  1.051705e+02  9.500337e-08  1.909630e-06  2.572247e-04 -3.121905e-05 
##            d2       file_id      fsc_perp     fsc_small            pe 
## -2.213118e-05  2.755981e-02  7.788155e-05  1.237508e-04 -8.244714e-05 
##          time 
##  1.971917e-04 
## [1] "class: pico:"
##                        .pos        .rnorm     chl_small            d1 
##  3.483912e+02 -9.441323e-07  2.590738e-03 -1.325554e-04  2.356812e-05 
##            d2       file_id       fsc_big      fsc_perp     fsc_small 
##  2.316151e-05 -7.045651e-02 -6.130446e-03 -5.924043e-05 -1.057377e-04 
##            pe          time 
## -2.546790e-04 -5.296027e-04 
## [1] "class: synecho:"
##                   chl_small      fsc_perp     fsc_small            pe 
##  1.328271e+02 -2.284018e-04 -2.724916e-05 -6.588735e-05  3.178569e-04 
## [1] "class: ultra:"
##                   chl_small            d2       file_id       fsc_big 
##  2.194298e+02  6.043786e-05 -1.385530e-06  1.004731e-02 -2.973058e-03 
##      fsc_perp     fsc_small            pe 
## -1.308556e-05  4.761619e-06 -1.025662e-04
```

```
## [1] TRUE
```

```r
# From here to save(), this should all be in one function
#   these are executed in the same seq twice more:
#       fit.data.training & predict.data.new chunks
print(sprintf("%s fit prediction diagnostics:", glb_sel_mdl_id))
```

```
## [1] "All.X###glmnet fit prediction diagnostics:"
```

```r
glbObsFit <- glb_get_predictions(df = glbObsFit, mdl_id = glb_sel_mdl_id, 
                                 rsp_var = glb_rsp_var)
print(sprintf("%s OOB prediction diagnostics:", glb_sel_mdl_id))
```

```
## [1] "All.X###glmnet OOB prediction diagnostics:"
```

```r
glbObsOOB <- glb_get_predictions(df = glbObsOOB, mdl_id = glb_sel_mdl_id, 
                                     rsp_var = glb_rsp_var)

print(glb_featsimp_df <- myget_feats_importance(mdl = glb_sel_mdl, featsimp_df = NULL))
```

```
##           All.X...glmnet.imp.crypto All.X...glmnet.imp.nano
## .pos                       71.25581                71.25591
## .rnorm                     71.25581                71.25581
## cell_id                    71.25581                71.25773
## chl_small                  71.28856                71.51482
## d1                         71.26464                71.22409
## d2                         71.26731                71.23331
## file_id                    71.25581                98.63535
## fsc_big                   100.00000                71.25581
## fsc_perp                   71.29304                71.33496
## fsc_small                  71.28422                71.38090
## pe                         71.39564                71.17275
## time                       71.25581                71.45262
##           All.X...glmnet.imp.pico All.X...glmnet.imp.synecho
## .pos                     71.25481                   71.25581
## .rnorm                   73.77245                   71.25581
## cell_id                  71.25581                   71.25581
## chl_small                71.12234                   71.02547
## d1                       71.27978                   71.25581
## d2                       71.27942                   71.25581
## file_id                   0.00000                   71.25581
## fsc_big                  65.00687                   71.25581
## fsc_perp                 71.19554                   71.22825
## fsc_small                71.14890                   71.18930
## pe                       70.99887                   71.57685
## time                     70.71917                   71.25581
##           All.X...glmnet.imp.ultra imp All.X###glmnet.imp
## .pos                      71.25581  -1                 -1
## .rnorm                    71.25581  -2                 -2
## cell_id                   71.25581  -3                 -3
## chl_small                 71.31703  -4                 -4
## d1                        71.25581  -5                 -5
## d2                        71.25453  -6                 -6
## file_id                   80.87732  -7                 -7
## fsc_big                   68.22399  -8                 -8
## fsc_perp                  71.24265  -9                 -9
## fsc_small                 71.26054 -10                -10
## pe                        71.15222 -11                -11
## time                      71.25581 -12                -12
```

```r
#mdl_id <-"RFE.X.glmnet"; glb_featsimp_df <- myget_feats_importance(glb_models_lst[[mdl_id]], glb_featsimp_df); glb_featsimp_df[, paste0(mdl_id, ".imp")] <- glb_featsimp_df$imp; print(glb_featsimp_df)
#print(head(sbst_featsimp_df <- subset(glb_featsimp_df, is.na(RFE.X.glmnet.imp) | (abs(RFE.X.YeoJohnson.glmnet.imp - RFE.X.glmnet.imp) > 0.0001), select=-imp)))
#print(orderBy(~ -cor.y.abs, subset(glb_feats_df, id %in% c(row.names(sbst_featsimp_df), "startprice.dcm1.is9", "D.weight.post.stop.sum"))))

# Used again in fit.data.training & predict.data.new chunks
glb_analytics_diag_plots <- function(obs_df, mdl_id, prob_threshold=NULL) {
    if (!is.null(featsimp_df <- glb_featsimp_df)) {
        featsimp_df$feat <- gsub("`(.*?)`", "\\1", row.names(featsimp_df))    
        featsimp_df$feat.interact <- gsub("(.*?):(.*)", "\\2", featsimp_df$feat)
        featsimp_df$feat <- gsub("(.*?):(.*)", "\\1", featsimp_df$feat)    
        featsimp_df$feat.interact <- 
            ifelse(featsimp_df$feat.interact == featsimp_df$feat, 
                                            NA, featsimp_df$feat.interact)
        featsimp_df$feat <- 
            gsub("(.*?)\\.fctr(.*)", "\\1\\.fctr", featsimp_df$feat)
        featsimp_df$feat.interact <- 
            gsub("(.*?)\\.fctr(.*)", "\\1\\.fctr", featsimp_df$feat.interact) 
        featsimp_df <- orderBy(~ -imp.max, 
            summaryBy(imp ~ feat + feat.interact, data=featsimp_df,
                      FUN=max))    
        #rex_str=":(.*)"; txt_vctr=tail(featsimp_df$feat); ret_lst <- regexec(rex_str, txt_vctr); ret_lst <- regmatches(txt_vctr, ret_lst); ret_vctr <- sapply(1:length(ret_lst), function(pos_ix) ifelse(length(ret_lst[[pos_ix]]) > 0, ret_lst[[pos_ix]], "")); print(ret_vctr <- ret_vctr[ret_vctr != ""])    
        
        featsimp_df <- subset(featsimp_df, !is.na(imp.max))
        if (nrow(featsimp_df) > 5) {
            warning("Limiting important feature scatter plots to 5 out of ",
                    nrow(featsimp_df))
            featsimp_df <- head(featsimp_df, 5)
        }
        
    #     if (!all(is.na(featsimp_df$feat.interact)))
    #         stop("not implemented yet")
        rsp_var_out <- mygetPredictIds(glb_rsp_var, mdl_id)$value
        for (var in featsimp_df$feat) {
            plot_df <- melt(obs_df, id.vars = var, 
                            measure.vars = c(glb_rsp_var, rsp_var_out))
    
            print(myplot_scatter(plot_df, var, "value", colorcol_name = "variable",
                                facet_colcol_name = "variable", jitter = TRUE) + 
                          guides(color = FALSE))
        }
    }
    
    if (glb_is_regression) {
        if (is.null(featsimp_df) || (nrow(featsimp_df) == 0))
            warning("No important features in glb_fin_mdl") else
            print(myplot_prediction_regression(df=obs_df, 
                        feat_x=ifelse(nrow(featsimp_df) > 1, featsimp_df$feat[2],
                                      ".rownames"), 
                                               feat_y=featsimp_df$feat[1],
                        rsp_var=glb_rsp_var, rsp_var_out=rsp_var_out,
                        id_vars=glb_id_var)
    #               + facet_wrap(reformulate(featsimp_df$feat[2])) # if [1 or 2] is a factor
    #               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
                  )
    }    
    
    if (glb_is_classification) {
        if (is.null(featsimp_df) || (nrow(featsimp_df) == 0))
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df = obs_df, 
                                feat_x = ifelse(nrow(featsimp_df) > 1, 
                                                featsimp_df$feat[2], ".rownames"),
                                               feat_y = featsimp_df$feat[1],
                                                rsp_var = glb_rsp_var, 
                                                rsp_var_out = rsp_var_out, 
                                                id_vars = glb_id_var,
                                                prob_threshold = prob_threshold))
    }    
}

if (glb_is_classification && glb_is_binomial)
    glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id, 
            prob_threshold = glb_models_df[glb_models_df$id == glb_sel_mdl_id, 
                                           "opt.prob.threshold.OOB"]) else
    glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id)                  
```

```
## Warning in glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id =
## glb_sel_mdl_id): Limiting important feature scatter plots to 5 out of 12
```

![](seaflow_base_files/figure-html/fit.models_2-9.png) ![](seaflow_base_files/figure-html/fit.models_2-10.png) ![](seaflow_base_files/figure-html/fit.models_2-11.png) ![](seaflow_base_files/figure-html/fit.models_2-12.png) ![](seaflow_base_files/figure-html/fit.models_2-13.png) 

```
## [1] "Min/Max Boundaries: "
##   .rownames pop.fctr pop.fctr.All.X...glmnet pop.fctr.All.X...glmnet.prob
## 1         2    ultra                   ultra                    0.6749597
## 2     26860     pico                    pico                    0.6671968
## 3     27479  synecho                 synecho                    0.9970029
## 4     72343  synecho                 synecho                    0.9971192
##   pop.fctr.All.X...glmnet.err pop.fctr.All.X...glmnet.err.abs
## 1                       FALSE                               0
## 2                       FALSE                               0
## 3                       FALSE                               0
## 4                       FALSE                               0
##   pop.fctr.All.X...glmnet.is.acc pop.fctr.All.X...glmnet.accurate
## 1                           TRUE                             TRUE
## 2                           TRUE                             TRUE
## 3                           TRUE                             TRUE
## 4                           TRUE                             TRUE
##   pop.fctr.All.X...glmnet.error .label
## 1                             0      2
## 2                             0  26860
## 3                             0  27479
## 4                             0  72343
## [1] "Inaccurate: "
##   .rownames pop.fctr pop.fctr.All.X...glmnet pop.fctr.All.X...glmnet.prob
## 1     62460   crypto                 synecho                    0.9996449
## 2     58296   crypto                 synecho                    0.9995949
## 3     36470    ultra                 synecho                    0.9995330
## 4     60290   crypto                 synecho                    0.9993987
## 5     68935   crypto                 synecho                    0.9988256
## 6     23448   crypto                 synecho                    0.9987388
##   pop.fctr.All.X...glmnet.err pop.fctr.All.X...glmnet.err.abs
## 1                        TRUE                    2.927242e-04
## 2                        TRUE                    3.593744e-04
## 3                        TRUE                    7.066188e-07
## 4                        TRUE                    4.973384e-04
## 5                        TRUE                    1.073863e-03
## 6                        TRUE                    8.484546e-04
##   pop.fctr.All.X...glmnet.is.acc pop.fctr.All.X...glmnet.accurate
## 1                          FALSE                            FALSE
## 2                          FALSE                            FALSE
## 3                          FALSE                            FALSE
## 4                          FALSE                            FALSE
## 5                          FALSE                            FALSE
## 6                          FALSE                            FALSE
##   pop.fctr.All.X...glmnet.error
## 1                  0.0003551254
## 2                  0.0004050600
## 3                  0.0004670371
## 4                  0.0006012996
## 5                  0.0011743566
## 6                  0.0012611745
##      .rownames pop.fctr pop.fctr.All.X...glmnet
## 253      53647    ultra                    pico
## 567      65452     pico                   ultra
## 1612     59277    ultra                    nano
## 2910     46078     pico                   ultra
## 3075     55690     nano                   ultra
## 5235     11361  synecho                   ultra
##      pop.fctr.All.X...glmnet.prob pop.fctr.All.X...glmnet.err
## 253                     0.7863506                        TRUE
## 567                     0.7313239                        TRUE
## 1612                    0.6423951                        TRUE
## 2910                    0.5681391                        TRUE
## 3075                    0.5620381                        TRUE
## 5235                    0.3778722                        TRUE
##      pop.fctr.All.X...glmnet.err.abs pop.fctr.All.X...glmnet.is.acc
## 253                        0.1913771                          FALSE
## 567                        0.1534413                          FALSE
## 1612                       0.2983456                          FALSE
## 2910                       0.3668833                          FALSE
## 3075                       0.4071246                          FALSE
## 5235                       0.3689951                          FALSE
##      pop.fctr.All.X...glmnet.accurate pop.fctr.All.X...glmnet.error
## 253                             FALSE                     0.2136494
## 567                             FALSE                     0.2686761
## 1612                            FALSE                     0.3576049
## 2910                            FALSE                     0.4318609
## 3075                            FALSE                     0.4379619
## 5235                            FALSE                     0.6221278
##      .rownames pop.fctr pop.fctr.All.X...glmnet
## 5232     39253     pico                 synecho
## 5233     71643     pico                   ultra
## 5234     22501     pico                   ultra
## 5235     11361  synecho                   ultra
## 5236     31634  synecho                   ultra
## 5237     29734     pico                 synecho
##      pop.fctr.All.X...glmnet.prob pop.fctr.All.X...glmnet.err
## 5232                    0.3877014                        TRUE
## 5233                    0.3854953                        TRUE
## 5234                    0.3792993                        TRUE
## 5235                    0.3778722                        TRUE
## 5236                    0.3635894                        TRUE
## 5237                    0.3408905                        TRUE
##      pop.fctr.All.X...glmnet.err.abs pop.fctr.All.X...glmnet.is.acc
## 5232                       0.3285955                          FALSE
## 5233                       0.3635779                          FALSE
## 5234                       0.3740928                          FALSE
## 5235                       0.3689951                          FALSE
## 5236                       0.3238271                          FALSE
## 5237                       0.2914950                          FALSE
##      pop.fctr.All.X...glmnet.accurate pop.fctr.All.X...glmnet.error
## 5232                            FALSE                     0.6122986
## 5233                            FALSE                     0.6145047
## 5234                            FALSE                     0.6207007
## 5235                            FALSE                     0.6221278
## 5236                            FALSE                     0.6364106
## 5237                            FALSE                     0.6591095
```

![](seaflow_base_files/figure-html/fit.models_2-14.png) 

```r
if (!is.null(glbFeatsCategory)) {
    glbLvlCategory <- merge(glbLvlCategory, 
            myget_category_stats(obs_df = glbObsFit, mdl_id = glb_sel_mdl_id, 
                                 label = "fit"), 
                            by = glbFeatsCategory, all = TRUE)
    row.names(glbLvlCategory) <- glbLvlCategory[, glbFeatsCategory]
    glbLvlCategory <- merge(glbLvlCategory, 
            myget_category_stats(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id,
                                 label="OOB"),
                          #by=glbFeatsCategory, all=TRUE) glb_ctgry-df already contains .n.OOB ?
                          all = TRUE)
    row.names(glbLvlCategory) <- glbLvlCategory[, glbFeatsCategory]
    if (any(grepl("OOB", glbMdlMetricsEval)))
        print(orderBy(~-err.abs.OOB.mean, glbLvlCategory)) else
            print(orderBy(~-err.abs.fit.mean, glbLvlCategory))
    print(colSums(glbLvlCategory[, -grep(glbFeatsCategory, names(glbLvlCategory))]))
}
```

```
##        .category .n.OOB .n.Fit .n.Tst .freqRatio.Fit .freqRatio.OOB
## .dummy    .dummy  36172  36171  36172              1              1
##        .freqRatio.Tst err.abs.fit.sum err.abs.fit.mean .n.fit
## .dummy              1        1804.883       0.04989861  36171
##        err.abs.OOB.sum err.abs.OOB.mean
## .dummy        1826.077       0.05048317
##           .n.OOB           .n.Fit           .n.Tst   .freqRatio.Fit 
##     3.617200e+04     3.617100e+04     3.617200e+04     1.000000e+00 
##   .freqRatio.OOB   .freqRatio.Tst  err.abs.fit.sum err.abs.fit.mean 
##     1.000000e+00     1.000000e+00     1.804883e+03     4.989861e-02 
##           .n.fit  err.abs.OOB.sum err.abs.OOB.mean 
##     3.617100e+04     1.826077e+03     5.048317e-02
```

```r
write.csv(glbObsOOB[, c(glb_id_var, 
                grep(glb_rsp_var, names(glbObsOOB), fixed=TRUE, value=TRUE))], 
    paste0(gsub(".", "_", paste0(glb_out_pfx, glb_sel_mdl_id), fixed=TRUE), 
           "_OOBobs.csv"), row.names=FALSE)

fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "teardown")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_2_bgn          1          0    teardown 684.698  NA      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc=FALSE)
```

```
##         label step_major step_minor label_minor     bgn    end elapsed
## 12 fit.models          6          2           2 646.152 684.73  38.578
## 13 fit.models          6          3           3 684.731     NA      NA
```


```r
# if (sum(is.na(glbObsAll$D.P.http)) > 0)
#         stop("fit.models_3: Why is this happening ?")

#stop(here"); glb2Sav()
sync_glb_obs_df <- function() {
    # Merge or cbind ?
    for (col in setdiff(names(glbObsFit), names(glbObsTrn)))
        glbObsTrn[glbObsTrn$.lcn == "Fit", col] <<- glbObsFit[, col]
    for (col in setdiff(names(glbObsFit), names(glbObsAll)))
        glbObsAll[glbObsAll$.lcn == "Fit", col] <<- glbObsFit[, col]
    if (all(is.na(glbObsNew[, glb_rsp_var])))
        for (col in setdiff(names(glbObsOOB), names(glbObsTrn)))
            glbObsTrn[glbObsTrn$.lcn == "OOB", col] <<- glbObsOOB[, col]
    for (col in setdiff(names(glbObsOOB), names(glbObsAll)))
        glbObsAll[glbObsAll$.lcn == "OOB", col] <<- glbObsOOB[, col]
}
sync_glb_obs_df()
    
print(setdiff(names(glbObsNew), names(glbObsAll)))
```

```
## character(0)
```

```r
if (glb_save_envir)
    save(glb_feats_df, 
         glbObsAll, #glbObsTrn, glbObsFit, glbObsOOB, glbObsNew,
         glb_models_df, dsp_models_df, glb_models_lst, glb_sel_mdl, glb_sel_mdl_id,
         glb_model_type,
        file=paste0(glb_out_pfx, "selmdl_dsk.RData"))
#load(paste0(glb_out_pfx, "selmdl_dsk.RData"))

rm(ret_lst)
```

```
## Warning in rm(ret_lst): object 'ret_lst' not found
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](seaflow_base_files/figure-html/fit.models_3-1.png) 

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=TRUE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 13        fit.models          6          3           3 684.731 710.187
## 14 fit.data.training          7          0           0 710.188      NA
##    elapsed
## 13  25.456
## 14      NA
```

## Step `7.0: fit data training`

```r
#load(paste0(glb_inp_pfx, "dsk.RData"))

if (!is.null(glb_fin_mdl_id) && (glb_fin_mdl_id %in% names(glb_models_lst))) {
    warning("Final model same as user selected model")
    glb_fin_mdl <- glb_models_lst[[glb_fin_mdl_id]]
} else 
# if (nrow(glbObsFit) + length(glbObsFitOutliers) == nrow(glbObsTrn))
if (!all(is.na(glbObsNew[, glb_rsp_var])))
{    
    warning("Final model same as glb_sel_mdl_id")
    glb_fin_mdl_id <- paste0("Final.", glb_sel_mdl_id)
    glb_fin_mdl <- glb_sel_mdl
    glb_models_lst[[glb_fin_mdl_id]] <- glb_fin_mdl
} else {    
            if (grepl("RFE\\.X", names(glbMdlFamilies))) {
                indep_vars <- myadjust_interaction_feats(subset(glb_feats_df, 
                                                    !nzv & (exclude.as.feat != 1))[, "id"])
                rfe_trn_results <- 
                    myrun_rfe(glbObsTrn, indep_vars, glbRFESizes[["Final"]])
                if (!isTRUE(all.equal(sort(predictors(rfe_trn_results)),
                                      sort(predictors(rfe_fit_results))))) {
                    print("Diffs predictors(rfe_trn_results) vs. predictors(rfe_fit_results):")
                    print(setdiff(predictors(rfe_trn_results), predictors(rfe_fit_results)))
                    print("Diffs predictors(rfe_fit_results) vs. predictors(rfe_trn_results):")
                    print(setdiff(predictors(rfe_fit_results), predictors(rfe_trn_results)))
            }
        }
    # }    

    if (grepl("Ensemble", glb_sel_mdl_id)) {
        # Find which models are relevant
        mdlimp_df <- subset(myget_feats_importance(glb_sel_mdl), imp > 5)
        # Fit selected models on glbObsTrn
        for (mdl_id in gsub(".prob", "", 
gsub(mygetPredictIds(glb_rsp_var)$value, "", row.names(mdlimp_df), fixed = TRUE),
                            fixed = TRUE)) {
            mdl_id_components <- unlist(strsplit(mdl_id, "[.]"))
            mdlIdPfx <- paste0(c(head(mdl_id_components, -1), "Train"), 
                               collapse = ".")
            if (grepl("RFE\\.X\\.", mdlIdPfx)) 
                mdlIndepVars <- myadjust_interaction_feats(myextract_actual_feats(
                    predictors(rfe_trn_results))) else
                mdlIndepVars <- trim(unlist(
            strsplit(glb_models_df[glb_models_df$id == mdl_id, "feats"], "[,]")))
            ret_lst <- 
                myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                        id.prefix = mdlIdPfx, 
                        type = glb_model_type, tune.df = glbMdlTuneParams,
                        trainControl.method = "repeatedcv",
                        trainControl.number = glb_rcv_n_folds,
                        trainControl.repeats = glb_rcv_n_repeats,
                        trainControl.classProbs = glb_is_classification,
                        trainControl.summaryFunction = glbMdlMetricSummaryFn,
                        train.metric = glbMdlMetricSummary, 
                        train.maximize = glbMdlMetricMaximize,    
                        train.method = tail(mdl_id_components, 1))),
                    indep_vars = mdlIndepVars,
                    rsp_var = glb_rsp_var, 
                    fit_df = glbObsTrn, OOB_df = NULL)
            
            glbObsTrn <- glb_get_predictions(df = glbObsTrn,
                                                mdl_id = tail(glb_models_df$id, 1), 
                                                rsp_var = glb_rsp_var,
                                                prob_threshold_def = 
                    subset(glb_models_df, id == mdl_id)$opt.prob.threshold.OOB)
            glbObsNew <- glb_get_predictions(df = glbObsNew,
                                                mdl_id = tail(glb_models_df$id, 1), 
                                                rsp_var = glb_rsp_var,
                                                prob_threshold_def = 
                    subset(glb_models_df, id == mdl_id)$opt.prob.threshold.OOB)
        }    
    }
    
    # "Final" model
    if ((model_method <- glb_sel_mdl$method) == "custom")
        # get actual method from the mdl_id
        model_method <- tail(unlist(strsplit(glb_sel_mdl_id, "[.]")), 1)
        
    if (grepl("Ensemble", glb_sel_mdl_id)) {
        # Find which models are relevant
        mdlimp_df <- subset(myget_feats_importance(glb_sel_mdl), imp > 5)
        if (glb_is_classification && glb_is_binomial)
            indep_vars_vctr <- gsub("(.*)\\.(.*)\\.prob", "\\1\\.Train\\.\\2\\.prob",
                                    row.names(mdlimp_df)) else
            indep_vars_vctr <- gsub("(.*)\\.(.*)", "\\1\\.Train\\.\\2",
                                    row.names(mdlimp_df))
    } else 
    if (grepl("RFE.X", glb_sel_mdl_id, fixed = TRUE)) {
        indep_vars_vctr <- myextract_actual_feats(predictors(rfe_trn_results))
    } else indep_vars_vctr <- 
                trim(unlist(strsplit(glb_models_df[glb_models_df$id ==
                                                   glb_sel_mdl_id
                                                   , "feats"], "[,]")))
        
    if (!is.null(glb_preproc_methods) &&
        ((match_pos <- regexpr(gsub(".", "\\.", 
                                    paste(glb_preproc_methods, collapse = "|"),
                                   fixed = TRUE), glb_sel_mdl_id)) != -1))
        ths_preProcess <- str_sub(glb_sel_mdl_id, match_pos, 
                                match_pos + attr(match_pos, "match.length") - 1) else
        ths_preProcess <- NULL                                      

    mdl_id_pfx <- ifelse(grepl("Ensemble", glb_sel_mdl_id),
                                   "Final.Ensemble", "Final")
    trnobs_df <- if (is.null(glbObsTrnOutliers[[mdl_id_pfx]])) glbObsTrn else 
        glbObsTrn[!(glbObsTrn[, glb_id_var] %in%
                            glbObsTrnOutliers[[mdl_id_pfx]]), ]
        
    # Force fitting of Final.glm to identify outliers
    method_vctr <- unique(c(myparseMdlId(glb_sel_mdl_id)$alg, glbMdlFamilies[["Final"]]))
    for (method in method_vctr) {
        #source("caret_nominalTrainWorkflow.R")
        
        # glmnet requires at least 2 indep vars
        if ((length(indep_vars_vctr) == 1) && (method %in% "glmnet"))
            next
        
        ret_lst <- 
            myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                    id.prefix = mdl_id_pfx, 
                    type = glb_model_type, trainControl.method = "repeatedcv",
                    trainControl.number = glb_rcv_n_folds, 
                    trainControl.repeats = glb_rcv_n_repeats,
                    trainControl.classProbs = glb_is_classification,
                    trainControl.summaryFunction = glbMdlMetricSummaryFn,
                    trainControl.allowParallel = glbMdlAllowParallel,
                    train.metric = glbMdlMetricSummary, 
                    train.maximize = glbMdlMetricMaximize,    
                    train.method = method,
                    train.preProcess = ths_preProcess)),
                indep_vars = indep_vars_vctr, rsp_var = glb_rsp_var, 
                fit_df = trnobs_df, OOB_df = NULL)
    }
        
    if ((length(method_vctr) == 1) || (method != "glm")) {
        glb_fin_mdl <- glb_models_lst[[length(glb_models_lst)]] 
        glb_fin_mdl_id <- glb_models_df[length(glb_models_lst), "id"]
    }
}
```

```
## Warning: Final model same as glb_sel_mdl_id
```

```r
rm(ret_lst)
```

```
## Warning in rm(ret_lst): object 'ret_lst' not found
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=FALSE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 14 fit.data.training          7          0           0 710.188 710.661
## 15 fit.data.training          7          1           1 710.661      NA
##    elapsed
## 14   0.473
## 15      NA
```


```r
#stop(here"); glb2Sav()
if (glb_is_classification && glb_is_binomial) 
    prob_threshold <- glb_models_df[glb_models_df$id == glb_sel_mdl_id,
                                        "opt.prob.threshold.OOB"] else 
    prob_threshold <- NULL

if (grepl("Ensemble", glb_fin_mdl_id)) {
    # Get predictions for each model in ensemble; Outliers that have been moved to OOB might not have been predicted yet
    mdlEnsembleComps <- unlist(str_split(subset(glb_models_df, 
                                                id == glb_fin_mdl_id)$feats, ","))
    if (glb_is_classification && glb_is_binomial)
        mdlEnsembleComps <- gsub("\\.prob$", "", mdlEnsembleComps)
    mdlEnsembleComps <- gsub(paste0("^", 
                        gsub(".", "\\.", mygetPredictIds(glb_rsp_var)$value, fixed = TRUE)),
                             "", mdlEnsembleComps)
    for (mdl_id in mdlEnsembleComps) {
        glbObsTrn <- glb_get_predictions(df = glbObsTrn, mdl_id = mdl_id, 
                                            rsp_var = glb_rsp_var,
                                            prob_threshold_def = prob_threshold)
        glbObsNew <- glb_get_predictions(df = glbObsNew, mdl_id = mdl_id, 
                                            rsp_var = glb_rsp_var,
                                            prob_threshold_def = prob_threshold)
    }    
}
glbObsTrn <- glb_get_predictions(df = glbObsTrn, mdl_id = glb_fin_mdl_id, 
                                     rsp_var = glb_rsp_var,
                                    prob_threshold_def = prob_threshold)

glb_featsimp_df <- myget_feats_importance(mdl=glb_fin_mdl,
                                          featsimp_df=glb_featsimp_df)
glb_featsimp_df[, paste0(glb_fin_mdl_id, ".imp")] <- glb_featsimp_df$imp
print(glb_featsimp_df)
```

```
##           All.X...glmnet.imp.crypto.x All.X...glmnet.imp.nano.x
## .pos                         71.25581                  71.25591
## .rnorm                       71.25581                  71.25581
## cell_id                      71.25581                  71.25773
## chl_small                    71.28856                  71.51482
## d1                           71.26464                  71.22409
## d2                           71.26731                  71.23331
## file_id                      71.25581                  98.63535
## fsc_big                     100.00000                  71.25581
## fsc_perp                     71.29304                  71.33496
## fsc_small                    71.28422                  71.38090
## pe                           71.39564                  71.17275
## time                         71.25581                  71.45262
##           All.X...glmnet.imp.pico.x All.X...glmnet.imp.synecho.x
## .pos                       71.25481                     71.25581
## .rnorm                     73.77245                     71.25581
## cell_id                    71.25581                     71.25581
## chl_small                  71.12234                     71.02547
## d1                         71.27978                     71.25581
## d2                         71.27942                     71.25581
## file_id                     0.00000                     71.25581
## fsc_big                    65.00687                     71.25581
## fsc_perp                   71.19554                     71.22825
## fsc_small                  71.14890                     71.18930
## pe                         70.99887                     71.57685
## time                       70.71917                     71.25581
##           All.X...glmnet.imp.ultra.x All.X###glmnet.imp
## .pos                        71.25581                 -1
## .rnorm                      71.25581                 -2
## cell_id                     71.25581                 -3
## chl_small                   71.31703                 -4
## d1                          71.25581                 -5
## d2                          71.25453                 -6
## file_id                     80.87732                 -7
## fsc_big                     68.22399                 -8
## fsc_perp                    71.24265                 -9
## fsc_small                   71.26054                -10
## pe                          71.15222                -11
## time                        71.25581                -12
##           All.X...glmnet.imp.crypto.y All.X...glmnet.imp.nano.y
## .pos                         71.25581                  71.25591
## .rnorm                       71.25581                  71.25581
## cell_id                      71.25581                  71.25773
## chl_small                    71.28856                  71.51482
## d1                           71.26464                  71.22409
## d2                           71.26731                  71.23331
## file_id                      71.25581                  98.63535
## fsc_big                     100.00000                  71.25581
## fsc_perp                     71.29304                  71.33496
## fsc_small                    71.28422                  71.38090
## pe                           71.39564                  71.17275
## time                         71.25581                  71.45262
##           All.X...glmnet.imp.pico.y All.X...glmnet.imp.synecho.y
## .pos                       71.25481                     71.25581
## .rnorm                     73.77245                     71.25581
## cell_id                    71.25581                     71.25581
## chl_small                  71.12234                     71.02547
## d1                         71.27978                     71.25581
## d2                         71.27942                     71.25581
## file_id                     0.00000                     71.25581
## fsc_big                    65.00687                     71.25581
## fsc_perp                   71.19554                     71.22825
## fsc_small                  71.14890                     71.18930
## pe                         70.99887                     71.57685
## time                       70.71917                     71.25581
##           All.X...glmnet.imp.ultra.y imp Final.All.X###glmnet.imp
## .pos                        71.25581  -1                       -1
## .rnorm                      71.25581  -2                       -2
## cell_id                     71.25581  -3                       -3
## chl_small                   71.31703  -4                       -4
## d1                          71.25581  -5                       -5
## d2                          71.25453  -6                       -6
## file_id                     80.87732  -7                       -7
## fsc_big                     68.22399  -8                       -8
## fsc_perp                    71.24265  -9                       -9
## fsc_small                   71.26054 -10                      -10
## pe                          71.15222 -11                      -11
## time                        71.25581 -12                      -12
```

```r
if (glb_is_classification && glb_is_binomial)
    glb_analytics_diag_plots(obs_df=glbObsTrn, mdl_id=glb_fin_mdl_id, 
            prob_threshold=glb_models_df[glb_models_df$id == glb_sel_mdl_id, 
                                         "opt.prob.threshold.OOB"]) else
    glb_analytics_diag_plots(obs_df=glbObsTrn, mdl_id=glb_fin_mdl_id)                  
```

```
## Warning in glb_analytics_diag_plots(obs_df = glbObsTrn, mdl_id =
## glb_fin_mdl_id): Limiting important feature scatter plots to 5 out of 12
```

![](seaflow_base_files/figure-html/fit.data.training_1-1.png) ![](seaflow_base_files/figure-html/fit.data.training_1-2.png) ![](seaflow_base_files/figure-html/fit.data.training_1-3.png) ![](seaflow_base_files/figure-html/fit.data.training_1-4.png) ![](seaflow_base_files/figure-html/fit.data.training_1-5.png) 

```
## [1] "Min/Max Boundaries: "
##   .rownames pop.fctr pop.fctr.All.X...glmnet pop.fctr.All.X...glmnet.prob
## 1         1     pico                    pico                    0.7619361
## 2     24121  synecho                 synecho                    0.9947160
## 3     66467  synecho                 synecho                    0.9918165
## 4     72341  synecho                 synecho                    0.9670190
##   pop.fctr.All.X...glmnet.err pop.fctr.All.X...glmnet.err.abs
## 1                       FALSE                               0
## 2                       FALSE                               0
## 3                       FALSE                               0
## 4                       FALSE                               0
##   pop.fctr.All.X...glmnet.is.acc pop.fctr.Final.All.X...glmnet
## 1                           TRUE                          pico
## 2                           TRUE                       synecho
## 3                           TRUE                       synecho
## 4                           TRUE                       synecho
##   pop.fctr.Final.All.X...glmnet.prob pop.fctr.Final.All.X...glmnet.err
## 1                          0.7619361                             FALSE
## 2                          0.9947160                             FALSE
## 3                          0.9918165                             FALSE
## 4                          0.9670190                             FALSE
##   pop.fctr.Final.All.X...glmnet.err.abs
## 1                                     0
## 2                                     0
## 3                                     0
## 4                                     0
##   pop.fctr.Final.All.X...glmnet.is.acc
## 1                                 TRUE
## 2                                 TRUE
## 3                                 TRUE
## 4                                 TRUE
##   pop.fctr.Final.All.X...glmnet.accurate
## 1                                   TRUE
## 2                                   TRUE
## 3                                   TRUE
## 4                                   TRUE
##   pop.fctr.Final.All.X...glmnet.error .label
## 1                                   0      1
## 2                                   0  24121
## 3                                   0  66467
## 4                                   0  72341
## [1] "Inaccurate: "
##   .rownames pop.fctr pop.fctr.All.X...glmnet pop.fctr.All.X...glmnet.prob
## 1     18071    ultra                 synecho                    0.9998599
## 2     65678     pico                 synecho                    0.9993263
## 3      7136     nano                 synecho                    0.9993091
## 4      4592    ultra                 synecho                    0.9992871
## 5     17335     nano                 synecho                    0.9991865
## 6        76   crypto                 synecho                    0.9961322
##   pop.fctr.All.X...glmnet.err pop.fctr.All.X...glmnet.err.abs
## 1                        TRUE                    4.026825e-07
## 2                        TRUE                    6.770284e-11
## 3                        TRUE                    4.133076e-06
## 4                        TRUE                    4.283866e-07
## 5                        TRUE                    1.589485e-05
## 6                        TRUE                    3.389558e-03
##   pop.fctr.All.X...glmnet.is.acc pop.fctr.Final.All.X...glmnet
## 1                          FALSE                       synecho
## 2                          FALSE                       synecho
## 3                          FALSE                       synecho
## 4                          FALSE                       synecho
## 5                          FALSE                       synecho
## 6                          FALSE                       synecho
##   pop.fctr.Final.All.X...glmnet.prob pop.fctr.Final.All.X...glmnet.err
## 1                          0.9998599                              TRUE
## 2                          0.9993263                              TRUE
## 3                          0.9993091                              TRUE
## 4                          0.9992871                              TRUE
## 5                          0.9991865                              TRUE
## 6                          0.9961322                              TRUE
##   pop.fctr.Final.All.X...glmnet.err.abs
## 1                          4.026825e-07
## 2                          6.770284e-11
## 3                          4.133076e-06
## 4                          4.283866e-07
## 5                          1.589485e-05
## 6                          3.389558e-03
##   pop.fctr.Final.All.X...glmnet.is.acc
## 1                                FALSE
## 2                                FALSE
## 3                                FALSE
## 4                                FALSE
## 5                                FALSE
## 6                                FALSE
##   pop.fctr.Final.All.X...glmnet.accurate
## 1                                  FALSE
## 2                                  FALSE
## 3                                  FALSE
## 4                                  FALSE
## 5                                  FALSE
## 6                                  FALSE
##   pop.fctr.Final.All.X...glmnet.error
## 1                        0.0001400548
## 2                        0.0006736953
## 3                        0.0006908841
## 4                        0.0007129266
## 5                        0.0008135085
## 6                        0.0038677941
##      .rownames pop.fctr pop.fctr.All.X...glmnet
## 655      63122    ultra                    nano
## 1660     12905    ultra                    nano
## 2319     70099   crypto                    nano
## 3657     61866     pico                   ultra
## 3699      7434     nano                   ultra
## 4444      8510    ultra                    pico
##      pop.fctr.All.X...glmnet.prob pop.fctr.All.X...glmnet.err
## 655                     0.7166353                        TRUE
## 1660                    0.6367064                        TRUE
## 2319                    0.5987301                        TRUE
## 3657                    0.5362868                        TRUE
## 3699                    0.5346488                        TRUE
## 4444                    0.5075577                        TRUE
##      pop.fctr.All.X...glmnet.err.abs pop.fctr.All.X...glmnet.is.acc
## 655                        0.2790891                          FALSE
## 1660                       0.3493456                          FALSE
## 2319                       0.3404287                          FALSE
## 3657                       0.4292840                          FALSE
## 3699                       0.4356683                          FALSE
## 4444                       0.4664800                          FALSE
##      pop.fctr.Final.All.X...glmnet pop.fctr.Final.All.X...glmnet.prob
## 655                           nano                          0.7166353
## 1660                          nano                          0.6367064
## 2319                          nano                          0.5987301
## 3657                         ultra                          0.5362868
## 3699                         ultra                          0.5346488
## 4444                          pico                          0.5075577
##      pop.fctr.Final.All.X...glmnet.err
## 655                               TRUE
## 1660                              TRUE
## 2319                              TRUE
## 3657                              TRUE
## 3699                              TRUE
## 4444                              TRUE
##      pop.fctr.Final.All.X...glmnet.err.abs
## 655                              0.2790891
## 1660                             0.3493456
## 2319                             0.3404287
## 3657                             0.4292840
## 3699                             0.4356683
## 4444                             0.4664800
##      pop.fctr.Final.All.X...glmnet.is.acc
## 655                                 FALSE
## 1660                                FALSE
## 2319                                FALSE
## 3657                                FALSE
## 3699                                FALSE
## 4444                                FALSE
##      pop.fctr.Final.All.X...glmnet.accurate
## 655                                   FALSE
## 1660                                  FALSE
## 2319                                  FALSE
## 3657                                  FALSE
## 3699                                  FALSE
## 4444                                  FALSE
##      pop.fctr.Final.All.X...glmnet.error
## 655                            0.2833647
## 1660                           0.3632936
## 2319                           0.4012699
## 3657                           0.4637132
## 3699                           0.4653512
## 4444                           0.4924423
##      .rownames pop.fctr pop.fctr.All.X...glmnet
## 5167     35158     pico                 synecho
## 5168     48469     pico                   ultra
## 5169     11257     pico                 synecho
## 5170     29218  synecho                    nano
## 5171     52145  synecho                    pico
## 5172     32345     pico                   ultra
##      pop.fctr.All.X...glmnet.prob pop.fctr.All.X...glmnet.err
## 5167                    0.3878718                        TRUE
## 5168                    0.3870044                        TRUE
## 5169                    0.3866755                        TRUE
## 5170                    0.3811510                        TRUE
## 5171                    0.3800630                        TRUE
## 5172                    0.3701962                        TRUE
##      pop.fctr.All.X...glmnet.err.abs pop.fctr.All.X...glmnet.is.acc
## 5167                       0.3455512                          FALSE
## 5168                       0.2221022                          FALSE
## 5169                       0.3535232                          FALSE
## 5170                       0.2200058                          FALSE
## 5171                       0.3037451                          FALSE
## 5172                       0.2665211                          FALSE
##      pop.fctr.Final.All.X...glmnet pop.fctr.Final.All.X...glmnet.prob
## 5167                       synecho                          0.3878718
## 5168                         ultra                          0.3870044
## 5169                       synecho                          0.3866755
## 5170                          nano                          0.3811510
## 5171                          pico                          0.3800630
## 5172                         ultra                          0.3701962
##      pop.fctr.Final.All.X...glmnet.err
## 5167                              TRUE
## 5168                              TRUE
## 5169                              TRUE
## 5170                              TRUE
## 5171                              TRUE
## 5172                              TRUE
##      pop.fctr.Final.All.X...glmnet.err.abs
## 5167                             0.3455512
## 5168                             0.2221022
## 5169                             0.3535232
## 5170                             0.2200058
## 5171                             0.3037451
## 5172                             0.2665211
##      pop.fctr.Final.All.X...glmnet.is.acc
## 5167                                FALSE
## 5168                                FALSE
## 5169                                FALSE
## 5170                                FALSE
## 5171                                FALSE
## 5172                                FALSE
##      pop.fctr.Final.All.X...glmnet.accurate
## 5167                                  FALSE
## 5168                                  FALSE
## 5169                                  FALSE
## 5170                                  FALSE
## 5171                                  FALSE
## 5172                                  FALSE
##      pop.fctr.Final.All.X...glmnet.error
## 5167                           0.6121282
## 5168                           0.6129956
## 5169                           0.6133245
## 5170                           0.6188490
## 5171                           0.6199370
## 5172                           0.6298038
```

![](seaflow_base_files/figure-html/fit.data.training_1-6.png) 

```r
dsp_feats_vctr <- c(NULL)
for(var in grep(".imp", names(glb_feats_df), fixed=TRUE, value=TRUE))
    dsp_feats_vctr <- union(dsp_feats_vctr, 
                            glb_feats_df[!is.na(glb_feats_df[, var]), "id"])

# print(glbObsTrn[glbObsTrn$UniqueID %in% FN_OOB_ids, 
#                     grep(glb_rsp_var, names(glbObsTrn), value=TRUE)])

print(setdiff(names(glbObsTrn), names(glbObsAll)))
```

```
## [1] "pop.fctr.Final.All.X...glmnet"        
## [2] "pop.fctr.Final.All.X...glmnet.prob"   
## [3] "pop.fctr.Final.All.X...glmnet.err"    
## [4] "pop.fctr.Final.All.X...glmnet.err.abs"
## [5] "pop.fctr.Final.All.X...glmnet.is.acc"
```

```r
for (col in setdiff(names(glbObsTrn), names(glbObsAll)))
    # Merge or cbind ?
    glbObsAll[glbObsAll$.src == "Train", col] <- glbObsTrn[, col]

print(setdiff(names(glbObsFit), names(glbObsAll)))
```

```
## character(0)
```

```r
print(setdiff(names(glbObsOOB), names(glbObsAll)))
```

```
## character(0)
```

```r
for (col in setdiff(names(glbObsOOB), names(glbObsAll)))
    # Merge or cbind ?
    glbObsAll[glbObsAll$.lcn == "OOB", col] <- glbObsOOB[, col]
    
print(setdiff(names(glbObsNew), names(glbObsAll)))
```

```
## character(0)
```

```r
if (glb_save_envir)
    save(glb_feats_df, glbObsAll, 
         #glbObsTrn, glbObsFit, glbObsOOB, glbObsNew,
         glb_models_df, dsp_models_df, glb_models_lst, glb_model_type,
         glb_sel_mdl, glb_sel_mdl_id,
         glb_fin_mdl, glb_fin_mdl_id,
        file=paste0(glb_out_pfx, "dsk.RData"))

replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](seaflow_base_files/figure-html/fit.data.training_1-7.png) 

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "predict.data.new", major.inc=TRUE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 15 fit.data.training          7          1           1 710.661 742.179
## 16  predict.data.new          8          0           0 742.179      NA
##    elapsed
## 15  31.518
## 16      NA
```

## Step `8.0: predict data new`

```
## Warning in glb_analytics_diag_plots(obs_df = glbObsNew, mdl_id =
## glb_fin_mdl_id, : Limiting important feature scatter plots to 5 out of 12
```

![](seaflow_base_files/figure-html/predict.data.new-1.png) ![](seaflow_base_files/figure-html/predict.data.new-2.png) ![](seaflow_base_files/figure-html/predict.data.new-3.png) ![](seaflow_base_files/figure-html/predict.data.new-4.png) ![](seaflow_base_files/figure-html/predict.data.new-5.png) 

```
## [1] "Min/Max Boundaries: "
##   .rownames pop.fctr pop.fctr.Final.All.X...glmnet
## 1         2    ultra                         ultra
## 2     26860     pico                          pico
## 3     27479  synecho                       synecho
## 4     72343  synecho                       synecho
##   pop.fctr.Final.All.X...glmnet.prob pop.fctr.Final.All.X...glmnet.err
## 1                          0.6749597                             FALSE
## 2                          0.6671968                             FALSE
## 3                          0.9970029                             FALSE
## 4                          0.9971192                             FALSE
##   pop.fctr.Final.All.X...glmnet.err.abs
## 1                                     0
## 2                                     0
## 3                                     0
## 4                                     0
##   pop.fctr.Final.All.X...glmnet.is.acc
## 1                                 TRUE
## 2                                 TRUE
## 3                                 TRUE
## 4                                 TRUE
##   pop.fctr.Final.All.X...glmnet.accurate
## 1                                   TRUE
## 2                                   TRUE
## 3                                   TRUE
## 4                                   TRUE
##   pop.fctr.Final.All.X...glmnet.error .label
## 1                                   0      2
## 2                                   0  26860
## 3                                   0  27479
## 4                                   0  72343
## [1] "Inaccurate: "
##   .rownames pop.fctr pop.fctr.Final.All.X...glmnet
## 1     62460   crypto                       synecho
## 2     58296   crypto                       synecho
## 3     36470    ultra                       synecho
## 4     60290   crypto                       synecho
## 5     68935   crypto                       synecho
## 6     23448   crypto                       synecho
##   pop.fctr.Final.All.X...glmnet.prob pop.fctr.Final.All.X...glmnet.err
## 1                          0.9996449                              TRUE
## 2                          0.9995949                              TRUE
## 3                          0.9995330                              TRUE
## 4                          0.9993987                              TRUE
## 5                          0.9988256                              TRUE
## 6                          0.9987388                              TRUE
##   pop.fctr.Final.All.X...glmnet.err.abs
## 1                          2.927242e-04
## 2                          3.593744e-04
## 3                          7.066188e-07
## 4                          4.973384e-04
## 5                          1.073863e-03
## 6                          8.484546e-04
##   pop.fctr.Final.All.X...glmnet.is.acc
## 1                                FALSE
## 2                                FALSE
## 3                                FALSE
## 4                                FALSE
## 5                                FALSE
## 6                                FALSE
##   pop.fctr.Final.All.X...glmnet.accurate
## 1                                  FALSE
## 2                                  FALSE
## 3                                  FALSE
## 4                                  FALSE
## 5                                  FALSE
## 6                                  FALSE
##   pop.fctr.Final.All.X...glmnet.error
## 1                        0.0003551254
## 2                        0.0004050600
## 3                        0.0004670371
## 4                        0.0006012996
## 5                        0.0011743566
## 6                        0.0012611745
##      .rownames pop.fctr pop.fctr.Final.All.X...glmnet
## 563      55307     nano                         ultra
## 829      71025     pico                         ultra
## 858      67895    ultra                          nano
## 2318     68966     pico                         ultra
## 4001     29262    ultra                          nano
## 4297     36037    ultra                          pico
##      pop.fctr.Final.All.X...glmnet.prob pop.fctr.Final.All.X...glmnet.err
## 563                           0.7323585                              TRUE
## 829                           0.7054456                              TRUE
## 858                           0.7022151                              TRUE
## 2318                          0.5983388                              TRUE
## 4001                          0.5237593                              TRUE
## 4297                          0.5127022                              TRUE
##      pop.fctr.Final.All.X...glmnet.err.abs
## 563                              0.2055346
## 829                              0.1904374
## 858                              0.2925377
## 2318                             0.3583512
## 4001                             0.4579210
## 4297                             0.4646276
##      pop.fctr.Final.All.X...glmnet.is.acc
## 563                                 FALSE
## 829                                 FALSE
## 858                                 FALSE
## 2318                                FALSE
## 4001                                FALSE
## 4297                                FALSE
##      pop.fctr.Final.All.X...glmnet.accurate
## 563                                   FALSE
## 829                                   FALSE
## 858                                   FALSE
## 2318                                  FALSE
## 4001                                  FALSE
## 4297                                  FALSE
##      pop.fctr.Final.All.X...glmnet.error
## 563                            0.2676415
## 829                            0.2945544
## 858                            0.2977849
## 2318                           0.4016612
## 4001                           0.4762407
## 4297                           0.4872978
##      .rownames pop.fctr pop.fctr.Final.All.X...glmnet
## 5232     39253     pico                       synecho
## 5233     71643     pico                         ultra
## 5234     22501     pico                         ultra
## 5235     11361  synecho                         ultra
## 5236     31634  synecho                         ultra
## 5237     29734     pico                       synecho
##      pop.fctr.Final.All.X...glmnet.prob pop.fctr.Final.All.X...glmnet.err
## 5232                          0.3877014                              TRUE
## 5233                          0.3854953                              TRUE
## 5234                          0.3792993                              TRUE
## 5235                          0.3778722                              TRUE
## 5236                          0.3635894                              TRUE
## 5237                          0.3408905                              TRUE
##      pop.fctr.Final.All.X...glmnet.err.abs
## 5232                             0.3285955
## 5233                             0.3635779
## 5234                             0.3740928
## 5235                             0.3689951
## 5236                             0.3238271
## 5237                             0.2914950
##      pop.fctr.Final.All.X...glmnet.is.acc
## 5232                                FALSE
## 5233                                FALSE
## 5234                                FALSE
## 5235                                FALSE
## 5236                                FALSE
## 5237                                FALSE
##      pop.fctr.Final.All.X...glmnet.accurate
## 5232                                  FALSE
## 5233                                  FALSE
## 5234                                  FALSE
## 5235                                  FALSE
## 5236                                  FALSE
## 5237                                  FALSE
##      pop.fctr.Final.All.X...glmnet.error
## 5232                           0.6122986
## 5233                           0.6145047
## 5234                           0.6207007
## 5235                           0.6221278
## 5236                           0.6364106
## 5237                           0.6591095
```

![](seaflow_base_files/figure-html/predict.data.new-6.png) 

```
## Loading required package: stringr
## Loading required package: tidyr
## 
## Attaching package: 'tidyr'
## 
## The following object is masked from 'package:Matrix':
## 
##     expand
```

```
## [1] "OOBobs pop.fctr.All.X...glmnet ultra: min < min of Train range: 6"
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id    d1
## 2             2                   ultra     2       4 12960
## 10455     10455                   ultra 10455       0 18016
## 10456     10456                   ultra 10456       4 20640
## 37482     37482                   ultra 37482       1 16608
## 52853     52853                   ultra 52853   10609  1536
## 60920     60920                   ultra 60920       3  5648
##              id      cor.y exclude.as.feat  cor.y.abs cor.high.X freqRatio
## .pos       .pos 0.04197727           FALSE 0.04197727       time  1.000000
## cell_id cell_id 0.00203730           FALSE 0.00203730       <NA>  1.000000
## d1           d1 0.06264683           FALSE 0.06264683       <NA>  1.079365
##         percentUnique zeroVar   nzv is.cor.y.abs.low interaction.feat
## .pos       100.000000   FALSE FALSE            FALSE               NA
## cell_id     61.919770   FALSE FALSE            FALSE               NA
## d1           6.673855   FALSE FALSE            FALSE               NA
##         shapiro.test.p.value rsp_var_raw rsp_var   max  min
## .pos            4.856422e-37       FALSE      NA 72343    1
## cell_id         2.576068e-35       FALSE      NA 32081    0
## d1              1.011239e-40       FALSE      NA 54048 1328
##         max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos                  71551             72319             72328
## cell_id               30290             32032             32043
## d1                    48960             53456             41456
##         max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                   72341              72335                  76
## cell_id                32035              32078                 112
## d1                     51712              45904                5472
##         min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                   48                 1                    5
## cell_id                 2                 0                    2
## d1                   1600              1328                 1600
##         min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                     4                              63492
## cell_id                  5                              17616
## d1                    1616                              50896
##         max.pop.fctr.All.X...glmnet.nano max.pop.fctr.All.X...glmnet.pico
## .pos                               72337                            72316
## cell_id                            32076                            31980
## d1                                 54048                            41808
##         max.pop.fctr.All.X...glmnet.synecho
## .pos                                  72343
## cell_id                               32053
## d1                                    53040
##         max.pop.fctr.All.X...glmnet.ultra
## .pos                                72342
## cell_id                             32081
## d1                                  43904
##         min.pop.fctr.All.X...glmnet.crypto
## .pos                                  6263
## cell_id                               2802
## d1                                   36496
##         min.pop.fctr.All.X...glmnet.nano min.pop.fctr.All.X...glmnet.pico
## .pos                                  15                                3
## cell_id                                1                                6
## d1                                  1792                             1936
##         min.pop.fctr.All.X...glmnet.synecho
## .pos                                     42
## cell_id                                   7
## d1                                     1488
##         min.pop.fctr.All.X...glmnet.ultra
## .pos                                    2
## cell_id                                 0
## d1                                   1536
##         max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                       63492
## cell_id                                    17616
## d1                                         50896
##         max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                     72337
## cell_id                                  32076
## d1                                       54048
##         max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                     72316
## cell_id                                  31980
## d1                                       41808
##         max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                        72343
## cell_id                                     32053
## d1                                          53040
##         max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                      72342
## cell_id                                   32081
## d1                                        43904
##         min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                        6263
## cell_id                                     2802
## d1                                         36496
##         min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                        15
## cell_id                                      1
## d1                                        1792
##         min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                         3
## cell_id                                      6
## d1                                        1936
##         min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                           42
## cell_id                                         7
## d1                                           1488
##         min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                          2
## cell_id                                       0
## d1                                         1536
## [1] "OOBobs pop.fctr.All.X...glmnet ultra: max > max of Train range: 8"
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id chl_small    d2
## 8233       8233                   ultra  8233   23305     25133 35952
## 23483     23483                   ultra 23483   11572     49184 22224
## 23543     23543                   ultra 23543   11752     26531 46016
## 52529     52529                   ultra 52529    9698     48869 27424
## 60919     60919                   ultra 60919   32081     33403 19888
## 66618     66618                   ultra 66618   15539     25595 35824
## 70887     70887                   ultra 70887   27173     50160 19120
## 72342     72342                   ultra 72342   31315     29435 18912
##       fsc_perp
## 8233     51483
## 23483    17824
## 23543    37005
## 52529    14291
## 60919    15885
## 66618    52669
## 70887    13416
## 72342    18301
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d2               d2  0.04550252           FALSE 0.04550252         d1
## fsc_perp   fsc_perp -0.12774069           FALSE 0.12774069       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos        1.00000    100.000000   FALSE FALSE            FALSE
## cell_id     1.00000     61.919770   FALSE FALSE            FALSE
## chl_small   1.00000     37.812059   FALSE FALSE            FALSE
## d2          1.02500      7.143845   FALSE FALSE            FALSE
## fsc_perp   21.58824     29.247187   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d2                      NA         8.244239e-28       FALSE      NA 54688
## fsc_perp                NA         2.470795e-26       FALSE      NA 63456
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos         1               71551             72319             72328
## cell_id      0               30290             32032             32043
## chl_small 3485               64784             64832             42016
## d2          32               54688             50960             43248
## fsc_perp     0               62173             55104             46880
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## chl_small                43307              47888               48888
## d2                       45792              45904                1712
## fsc_perp                 49589              51237                8528
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## chl_small             30251             12003                 3547
## d2                       48                32                   32
## fsc_perp               4563                 0                    0
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## chl_small              14485                              64789
## d2                        32                              49232
## fsc_perp                   0                              56579
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## chl_small                            64805
## d2                                   54096
## fsc_perp                             63456
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## chl_small                            40797
## d2                                   42064
## fsc_perp                             45952
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## chl_small                               64776
## d2                                      51872
## fsc_perp                                53571
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## chl_small                             50160
## d2                                    46016
## fsc_perp                              52669
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## chl_small                              38267
## d2                                     38048
## fsc_perp                               44157
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## chl_small                            27245
## d2                                      32
## fsc_perp                             12131
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## chl_small                             5672
## d2                                      32
## fsc_perp                                 0
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## chl_small                                3485
## d2                                         32
## fsc_perp                                    0
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## chl_small                             20725
## d2                                       32
## fsc_perp                               2160
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## chl_small                                    64789
## d2                                           49232
## fsc_perp                                     56579
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## chl_small                                  64805
## d2                                         54096
## fsc_perp                                   63456
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## chl_small                                  40797
## d2                                         42064
## fsc_perp                                   45952
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## chl_small                                     64776
## d2                                            51872
## fsc_perp                                      53571
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## chl_small                                   50160
## d2                                          46016
## fsc_perp                                    52669
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## chl_small                                    38267
## d2                                           38048
## fsc_perp                                     44157
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## chl_small                                  27245
## d2                                            32
## fsc_perp                                   12131
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## chl_small                                   5672
## d2                                            32
## fsc_perp                                       0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## chl_small                                      3485
## d2                                               32
## fsc_perp                                          0
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## chl_small                                   20725
## d2                                             32
## fsc_perp                                     2160
## [1] "OOBobs pop.fctr.All.X...glmnet pico: min < min of Train range: 26"
##      .rownames pop.fctr.All.X...glmnet chl_small fsc_small
## 396        396                    pico      5968     10232
## 824        824                    pico      7653     13984
## 1330      1330                    pico      6232     16581
## 1696      1696                    pico      6741     11712
## 2539      2539                    pico      6416     17803
## 2702      2702                    pico      5672     17885
##       .rownames pop.fctr.All.X...glmnet chl_small fsc_small
## 396         396                    pico      5968     10232
## 2702       2702                    pico      5672     17885
## 18205     18205                    pico     11053     18195
## 19056     19056                    pico     11504     12347
## 23242     23242                    pico     11827     33133
## 27339     27339                    pico      9003     19853
##       .rownames pop.fctr.All.X...glmnet chl_small fsc_small
## 23242     23242                    pico     11827     33133
## 27339     27339                    pico      9003     19853
## 27955     27955                    pico     10936     16475
## 31863     31863                    pico     14312     10005
## 32330     32330                    pico     11965     21280
## 60756     60756                    pico     12000     15115
##                  id      cor.y exclude.as.feat cor.y.abs cor.high.X
## chl_small chl_small -0.1960482           FALSE 0.1960482    chl_big
## fsc_small fsc_small -0.1175350           FALSE 0.1175350   fsc_perp
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small         1      37.81206   FALSE FALSE            FALSE
## fsc_small         1      30.47745   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small  3485               64784             64832             42016
## fsc_small 10005               65424             65365             53568
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## fsc_small                63547              56896               29088
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## fsc_small             24843             10011                10061
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## fsc_small              10376                              65405
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## fsc_small                            65424
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## fsc_small                            49208
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## fsc_small                               65416
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## fsc_small                             54885
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## fsc_small                              58219
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## fsc_small                            26208
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## fsc_small                            10005
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## fsc_small                               10056
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## fsc_small                             18821
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## fsc_small                                    65405
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## fsc_small                                  65424
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## fsc_small                                  49208
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## fsc_small                                     65416
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## fsc_small                                   54885
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## fsc_small                                    58219
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## fsc_small                                  26208
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## fsc_small                                  10005
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## fsc_small                                     10056
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## fsc_small                                   18821
## [1] "OOBobs pop.fctr.All.X...glmnet pico: max > max of Train range: 4"
##       .rownames pop.fctr.All.X...glmnet    d1 fsc_big
## 18348     18348                    pico 28544   32448
## 20094     20094                    pico 41808   32400
## 26156     26156                    pico  9264   32448
## 49143     49143                    pico 27728   32448
##              id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## d1           d1  0.06264683           FALSE 0.06264683       <NA>
## fsc_big fsc_big -0.03147761           FALSE 0.03147761       <NA>
##         freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## d1       1.079365    6.67385475   FALSE FALSE            FALSE
## fsc_big  2.153882    0.01658787   FALSE FALSE            FALSE
##         interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## d1                    NA         1.011239e-40       FALSE      NA 54048
## fsc_big               NA         3.294647e-68       FALSE      NA 32464
##           min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## d1       1328               48960             53456             41456
## fsc_big 32384               32448             32464             32432
##         max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## d1                     51712              45904                5472
## fsc_big                32464              32448               32416
##         min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## d1                   1600              1328                 1600
## fsc_big             32384             32384                32384
##         min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## d1                    1616                              50896
## fsc_big              32384                              32448
##         max.pop.fctr.All.X...glmnet.nano max.pop.fctr.All.X...glmnet.pico
## d1                                 54048                            41808
## fsc_big                            32448                            32448
##         max.pop.fctr.All.X...glmnet.synecho
## d1                                    53040
## fsc_big                               32448
##         max.pop.fctr.All.X...glmnet.ultra
## d1                                  43904
## fsc_big                             32448
##         min.pop.fctr.All.X...glmnet.crypto
## d1                                   36496
## fsc_big                              32432
##         min.pop.fctr.All.X...glmnet.nano min.pop.fctr.All.X...glmnet.pico
## d1                                  1792                             1936
## fsc_big                            32384                            32384
##         min.pop.fctr.All.X...glmnet.synecho
## d1                                     1488
## fsc_big                               32384
##         min.pop.fctr.All.X...glmnet.ultra
## d1                                   1536
## fsc_big                             32384
##         max.pop.fctr.Final.All.X...glmnet.crypto
## d1                                         50896
## fsc_big                                    32448
##         max.pop.fctr.Final.All.X...glmnet.nano
## d1                                       54048
## fsc_big                                  32448
##         max.pop.fctr.Final.All.X...glmnet.pico
## d1                                       41808
## fsc_big                                  32448
##         max.pop.fctr.Final.All.X...glmnet.synecho
## d1                                          53040
## fsc_big                                     32448
##         max.pop.fctr.Final.All.X...glmnet.ultra
## d1                                        43904
## fsc_big                                   32448
##         min.pop.fctr.Final.All.X...glmnet.crypto
## d1                                         36496
## fsc_big                                    32432
##         min.pop.fctr.Final.All.X...glmnet.nano
## d1                                        1792
## fsc_big                                  32384
##         min.pop.fctr.Final.All.X...glmnet.pico
## d1                                        1936
## fsc_big                                  32384
##         min.pop.fctr.Final.All.X...glmnet.synecho
## d1                                           1488
## fsc_big                                     32384
##         min.pop.fctr.Final.All.X...glmnet.ultra
## d1                                         1536
## fsc_big                                   32384
## [1] "OOBobs pop.fctr.All.X...glmnet nano: min < min of Train range: 9"
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id chl_small    d2 time
## 15           15                    nano    15      37     40597 28288   12
## 22           22                    nano    22      59     45275  7520   12
## 29           29                    nano    29      82     42499  6016   12
## 34           34                    nano    34      96     45800 24208   12
## 46           46                    nano    46     133     42077  8432   13
## 11405     11405                    nano 11405    2903     42696    32  111
## 19890     19890                    nano 19890       1     44509 31984  192
## 59277     59277                    nano 59277   27708     30013 44512  541
## 70195     70195                    nano 70195   25242     27245 20560  624
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d2               d2  0.04550252           FALSE 0.04550252         d1
## time           time  0.04275996           FALSE 0.04275996    file_id
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos          1.000    100.000000   FALSE FALSE            FALSE
## cell_id       1.000     61.919770   FALSE FALSE            FALSE
## chl_small     1.000     37.812059   FALSE FALSE            FALSE
## d2            1.025      7.143845   FALSE FALSE            FALSE
## time          1.040      1.747256   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d2                      NA         8.244239e-28       FALSE      NA 54688
## time                    NA         1.186295e-39       FALSE      NA   643
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos         1               71551             72319             72328
## cell_id      0               30290             32032             32043
## chl_small 3485               64784             64832             42016
## d2          32               54688             50960             43248
## time        12                 636               642               642
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## chl_small                43307              47888               48888
## d2                       45792              45904                1712
## time                       643                643                  13
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## chl_small             30251             12003                 3547
## d2                       48                32                   32
## time                     13                12                   12
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## chl_small              14485                              64789
## d2                        32                              49232
## time                      12                                572
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## chl_small                            64805
## d2                                   54096
## time                                   643
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## chl_small                            40797
## d2                                   42064
## time                                   642
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## chl_small                               64776
## d2                                      51872
## time                                      643
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## chl_small                             50160
## d2                                    46016
## time                                    643
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## chl_small                              38267
## d2                                     38048
## time                                      66
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## chl_small                            27245
## d2                                      32
## time                                    12
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## chl_small                             5672
## d2                                      32
## time                                    12
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## chl_small                                3485
## d2                                         32
## time                                       12
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## chl_small                             20725
## d2                                       32
## time                                     12
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## chl_small                                    64789
## d2                                           49232
## time                                           572
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## chl_small                                  64805
## d2                                         54096
## time                                         643
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## chl_small                                  40797
## d2                                         42064
## time                                         642
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## chl_small                                     64776
## d2                                            51872
## time                                            643
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## chl_small                                   50160
## d2                                          46016
## time                                          643
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## chl_small                                    38267
## d2                                           38048
## time                                            66
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## chl_small                                  27245
## d2                                            32
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## chl_small                                   5672
## d2                                            32
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## chl_small                                      3485
## d2                                               32
## time                                             12
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## chl_small                                   20725
## d2                                             32
## time                                           12
## [1] "OOBobs pop.fctr.All.X...glmnet nano: max > max of Train range: 19"
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id    d1    d2 fsc_perp
## 876         876                    nano   876    2321 47824 53056    63456
## 1718       1718                    nano  1718    4606 41520 46208    49587
## 2887       2887                    nano  2887    7880 39904 44032    53051
## 17371     17371                    nano 17371   20700 45648 54096    46536
## 30453     30453                    nano 30453    7125 54048 45056    45077
## 43815     43815                    nano 43815   17740 45456 50640    51181
## 44731     44731                    nano 44731   20093 31040 40080    53755
## 51111     51111                    nano 51111    6102 50320 47584    60403
## 51444     51444                    nano 51444    6906 40496 51856    47691
## 57830     57830                    nano 57830   23840 40384 51632    42419
## 59421     59421                    nano 59421   28081 43376 52192    43976
## 60917     60917                    nano 60917   32076 35680 36848    37659
## 64561     64561                    nano 64561    9802 47040 46800    58957
## 65616     65616                    nano 65616   12670 42064 45808    60195
## 70085     70085                    nano 70085   24927 45136 52784    53784
## 70897     70897                    nano 70897   27194 40624 53136    46573
## 72322     72322                    nano 72322   31258 10608 13328    33699
## 72325     72325                    nano 72325   31263 17328 15040    29789
## 72337     72337                    nano 72337   31303 12912 20448    27648
##       fsc_small    pe time
## 876       63640 20232   20
## 1718      63147 50731   26
## 2887      62832 53245   36
## 17371     63317 16133  168
## 30453     61456 17493  306
## 43815     65424 14637  424
## 44731     63171 54117  431
## 51111     64619 21979  480
## 51444     63971 20760  482
## 57830     57699  8936  530
## 59421     61837 13131  542
## 60917     56301  3859  553
## 64561     65397 24707  580
## 65616     65373 45571  588
## 70085     65341 25395  624
## 70897     59419 13741  630
## 72322     47848  2979  642
## 72325     47653  1171  642
## 72337     45987  2277  643
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## d2               d2  0.04550252           FALSE 0.04550252         d1
## fsc_perp   fsc_perp -0.12774069           FALSE 0.12774069       <NA>
## fsc_small fsc_small -0.11753499           FALSE 0.11753499   fsc_perp
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
## time           time  0.04275996           FALSE 0.04275996    file_id
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos       1.000000    100.000000   FALSE FALSE            FALSE
## cell_id    1.000000     61.919770   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## d2         1.025000      7.143845   FALSE FALSE            FALSE
## fsc_perp  21.588235     29.247187   FALSE FALSE            FALSE
## fsc_small  1.000000     30.477454   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
## time       1.040000      1.747256   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## d1                      NA         1.011239e-40       FALSE      NA 54048
## d2                      NA         8.244239e-28       FALSE      NA 54688
## fsc_perp                NA         2.470795e-26       FALSE      NA 63456
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
## pe                      NA         7.564684e-71       FALSE      NA 58675
## time                    NA         1.186295e-39       FALSE      NA   643
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos          1               71551             72319             72328
## cell_id       0               30290             32032             32043
## d1         1328               48960             53456             41456
## d2           32               54688             50960             43248
## fsc_perp      0               62173             55104             46880
## fsc_small 10005               65424             65365             53568
## pe            0               58267             50416             45107
## time         12                 636               642               642
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## d1                       51712              45904                5472
## d2                       45792              45904                1712
## fsc_perp                 49589              51237                8528
## fsc_small                63547              56896               29088
## pe                       43744              46688               31125
## time                       643                643                  13
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## d1                     1600              1328                 1600
## d2                       48                32                   32
## fsc_perp               4563                 0                    0
## fsc_small             24843             10011                10061
## pe                        0                 0                 5005
## time                     13                12                   12
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## d1                      1616                              50896
## d2                        32                              49232
## fsc_perp                   0                              56579
## fsc_small              10376                              65405
## pe                         0                              58675
## time                      12                                572
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## d1                                   54048
## d2                                   54096
## fsc_perp                             63456
## fsc_small                            65424
## pe                                   54117
## time                                   643
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## d1                                   41808
## d2                                   42064
## fsc_perp                             45952
## fsc_small                            49208
## pe                                    8091
## time                                   642
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## d1                                      53040
## d2                                      51872
## fsc_perp                                53571
## fsc_small                               65416
## pe                                      56280
## time                                      643
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## d1                                    43904
## d2                                    46016
## fsc_perp                              52669
## fsc_small                             54885
## pe                                    14963
## time                                    643
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## d1                                     36496
## d2                                     38048
## fsc_perp                               44157
## fsc_small                              58219
## pe                                     30813
## time                                      66
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## d1                                    1792
## d2                                      32
## fsc_perp                             12131
## fsc_small                            26208
## pe                                       0
## time                                    12
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## d1                                    1936
## d2                                      32
## fsc_perp                                 0
## fsc_small                            10005
## pe                                       0
## time                                    12
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## d1                                       1488
## d2                                         32
## fsc_perp                                    0
## fsc_small                               10056
## pe                                       2773
## time                                       12
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## d1                                     1536
## d2                                       32
## fsc_perp                               2160
## fsc_small                             18821
## pe                                        0
## time                                     12
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## d1                                           50896
## d2                                           49232
## fsc_perp                                     56579
## fsc_small                                    65405
## pe                                           58675
## time                                           572
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## d1                                         54048
## d2                                         54096
## fsc_perp                                   63456
## fsc_small                                  65424
## pe                                         54117
## time                                         643
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## d1                                         41808
## d2                                         42064
## fsc_perp                                   45952
## fsc_small                                  49208
## pe                                          8091
## time                                         642
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## d1                                            53040
## d2                                            51872
## fsc_perp                                      53571
## fsc_small                                     65416
## pe                                            56280
## time                                            643
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## d1                                          43904
## d2                                          46016
## fsc_perp                                    52669
## fsc_small                                   54885
## pe                                          14963
## time                                          643
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## d1                                           36496
## d2                                           38048
## fsc_perp                                     44157
## fsc_small                                    58219
## pe                                           30813
## time                                            66
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## d1                                          1792
## d2                                            32
## fsc_perp                                   12131
## fsc_small                                  26208
## pe                                             0
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## d1                                          1936
## d2                                            32
## fsc_perp                                       0
## fsc_small                                  10005
## pe                                             0
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## d1                                             1488
## d2                                               32
## fsc_perp                                          0
## fsc_small                                     10056
## pe                                             2773
## time                                             12
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## d1                                           1536
## d2                                             32
## fsc_perp                                     2160
## fsc_small                                   18821
## pe                                              0
## time                                           12
## [1] "OOBobs pop.fctr.All.X...glmnet synecho: min < min of Train range: 68"
##       .rownames pop.fctr.All.X...glmnet chl_small    d1 fsc_small   pe
## 1304       1304                 synecho      3675 14256     20483 5003
## 1614       1614                 synecho      3485  4224     15827 5168
## 3006       3006                 synecho     12989  6848     24248 4251
## 13772     13772                 synecho     18429  5696     37984 4624
## 17188     17188                 synecho     17067  8736     35368 4797
## 23562     23562                 synecho     16157  3936     26157 4949
##       .rownames pop.fctr.All.X...glmnet chl_small    d1 fsc_small   pe
## 23562     23562                 synecho     16157  3936     26157 4949
## 45772     45772                 synecho     17861  9152     39587 4603
## 52402     52402                 synecho     12307 20960     30040 4936
## 57338     57338                 synecho     12259 16144     29184 4581
## 67084     67084                 synecho     18379  6720     37120 3739
## 71638     71638                 synecho     15384 17984     31781 4819
##       .rownames pop.fctr.All.X...glmnet chl_small    d1 fsc_small   pe
## 69562     69562                 synecho     12896  8064     27181 4504
## 70211     70211                 synecho     12667 20800     33069 4357
## 71096     71096                 synecho     17800 22368     33277 4952
## 71353     71353                 synecho     14675 15552     29461 4544
## 71638     71638                 synecho     15384 17984     31781 4819
## 71996     71996                 synecho     14005 25328     34144 4453
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## fsc_small fsc_small -0.11753499           FALSE 0.11753499   fsc_perp
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small  1.000000     37.812059   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## fsc_small  1.000000     30.477454   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d1                      NA         1.011239e-40       FALSE      NA 54048
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
## pe                      NA         7.564684e-71       FALSE      NA 58675
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small  3485               64784             64832             42016
## d1         1328               48960             53456             41456
## fsc_small 10005               65424             65365             53568
## pe            0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## d1                       51712              45904                5472
## fsc_small                63547              56896               29088
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## d1                     1600              1328                 1600
## fsc_small             24843             10011                10061
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## d1                      1616                              50896
## fsc_small              10376                              65405
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## d1                                   54048
## fsc_small                            65424
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## d1                                   41808
## fsc_small                            49208
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## d1                                      53040
## fsc_small                               65416
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## d1                                    43904
## fsc_small                             54885
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## d1                                     36496
## fsc_small                              58219
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## d1                                    1792
## fsc_small                            26208
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## d1                                    1936
## fsc_small                            10005
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## d1                                       1488
## fsc_small                               10056
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## d1                                     1536
## fsc_small                             18821
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## d1                                           50896
## fsc_small                                    65405
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## d1                                         54048
## fsc_small                                  65424
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## d1                                         41808
## fsc_small                                  49208
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## d1                                            53040
## fsc_small                                     65416
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## d1                                          43904
## fsc_small                                   54885
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## d1                                           36496
## fsc_small                                    58219
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## d1                                          1792
## fsc_small                                  26208
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## d1                                          1936
## fsc_small                                  10005
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## d1                                             1488
## fsc_small                                     10056
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## d1                                           1536
## fsc_small                                   18821
## pe                                              0
## [1] "OOBobs pop.fctr.All.X...glmnet synecho: max > max of Train range: 28"
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id chl_small    d1
## 3519       3519                 synecho  3519    9643     44493 14160
## 5319       5319                 synecho  5319   14799     35125 37968
## 7991       7991                 synecho  7991   22556     54011  5520
## 9689       9689                 synecho  9689   27526     64776  7744
## 13776     13776                 synecho 13776    9824     43357  8032
## 23448     23448                 synecho 23448   11458     47072  7824
##          d2 fsc_perp fsc_small    pe
## 3519   7840     1035     17752 40971
## 5319  40400    38208     52387 50397
## 7991  14512    14056     29315 49925
## 9689  16160    27416     46179 55357
## 13776  4688    23056     38944 46443
## 23448  4672     8157     26499 38419
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id chl_small    d1
## 41892     41892                 synecho 41892   12463     43397 10160
## 45072     45072                 synecho 45072   20956     59827 38160
## 58296     58296                 synecho 58296   25066     45557 12000
## 60914     60914                 synecho 60914   32053     13752 29872
## 67647     67647                 synecho 67647   18280     44603 37200
## 68935     68935                 synecho 68935   21773     49792  6304
##          d2 fsc_perp fsc_small    pe
## 41892 13504    33669     50821 34755
## 45072 31856    28664     46411 47952
## 58296 22032     3003     22616 41003
## 60914 22608    10664     29051  5845
## 67647 47376    49253     63763 47880
## 68935   944    12099     32053 47595
##       .rownames pop.fctr.All.X...glmnet  .pos cell_id chl_small    d1
## 68935     68935                 synecho 68935   21773     49792  6304
## 69447     69447                 synecho 69447   23100     44331  3264
## 69623     69623                 synecho 69623   23581     60560 28624
## 71612     71612                 synecho 71612   29172     47803 42320
## 72034     72034                 synecho 72034   30430     36096 53040
## 72343     72343                 synecho 72343   31316     21288  6592
##          d2 fsc_perp fsc_small    pe
## 68935   944    12099     32053 47595
## 69447   448     1851     18643 25437
## 69623 28928    26795     46453 50059
## 71612 48976    53571     65416 50821
## 72034 48400    47851     62496 43957
## 72343  4704    22192     37096 17995
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## d2               d2  0.04550252           FALSE 0.04550252         d1
## fsc_perp   fsc_perp -0.12774069           FALSE 0.12774069       <NA>
## fsc_small fsc_small -0.11753499           FALSE 0.11753499   fsc_perp
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos       1.000000    100.000000   FALSE FALSE            FALSE
## cell_id    1.000000     61.919770   FALSE FALSE            FALSE
## chl_small  1.000000     37.812059   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## d2         1.025000      7.143845   FALSE FALSE            FALSE
## fsc_perp  21.588235     29.247187   FALSE FALSE            FALSE
## fsc_small  1.000000     30.477454   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d1                      NA         1.011239e-40       FALSE      NA 54048
## d2                      NA         8.244239e-28       FALSE      NA 54688
## fsc_perp                NA         2.470795e-26       FALSE      NA 63456
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
## pe                      NA         7.564684e-71       FALSE      NA 58675
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos          1               71551             72319             72328
## cell_id       0               30290             32032             32043
## chl_small  3485               64784             64832             42016
## d1         1328               48960             53456             41456
## d2           32               54688             50960             43248
## fsc_perp      0               62173             55104             46880
## fsc_small 10005               65424             65365             53568
## pe            0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## chl_small                43307              47888               48888
## d1                       51712              45904                5472
## d2                       45792              45904                1712
## fsc_perp                 49589              51237                8528
## fsc_small                63547              56896               29088
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## chl_small             30251             12003                 3547
## d1                     1600              1328                 1600
## d2                       48                32                   32
## fsc_perp               4563                 0                    0
## fsc_small             24843             10011                10061
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## chl_small              14485                              64789
## d1                      1616                              50896
## d2                        32                              49232
## fsc_perp                   0                              56579
## fsc_small              10376                              65405
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## chl_small                            64805
## d1                                   54048
## d2                                   54096
## fsc_perp                             63456
## fsc_small                            65424
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## chl_small                            40797
## d1                                   41808
## d2                                   42064
## fsc_perp                             45952
## fsc_small                            49208
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## chl_small                               64776
## d1                                      53040
## d2                                      51872
## fsc_perp                                53571
## fsc_small                               65416
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## chl_small                             50160
## d1                                    43904
## d2                                    46016
## fsc_perp                              52669
## fsc_small                             54885
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## chl_small                              38267
## d1                                     36496
## d2                                     38048
## fsc_perp                               44157
## fsc_small                              58219
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## chl_small                            27245
## d1                                    1792
## d2                                      32
## fsc_perp                             12131
## fsc_small                            26208
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## chl_small                             5672
## d1                                    1936
## d2                                      32
## fsc_perp                                 0
## fsc_small                            10005
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## chl_small                                3485
## d1                                       1488
## d2                                         32
## fsc_perp                                    0
## fsc_small                               10056
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## chl_small                             20725
## d1                                     1536
## d2                                       32
## fsc_perp                               2160
## fsc_small                             18821
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## chl_small                                    64789
## d1                                           50896
## d2                                           49232
## fsc_perp                                     56579
## fsc_small                                    65405
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## chl_small                                  64805
## d1                                         54048
## d2                                         54096
## fsc_perp                                   63456
## fsc_small                                  65424
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## chl_small                                  40797
## d1                                         41808
## d2                                         42064
## fsc_perp                                   45952
## fsc_small                                  49208
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## chl_small                                     64776
## d1                                            53040
## d2                                            51872
## fsc_perp                                      53571
## fsc_small                                     65416
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## chl_small                                   50160
## d1                                          43904
## d2                                          46016
## fsc_perp                                    52669
## fsc_small                                   54885
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## chl_small                                    38267
## d1                                           36496
## d2                                           38048
## fsc_perp                                     44157
## fsc_small                                    58219
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## chl_small                                  27245
## d1                                          1792
## d2                                            32
## fsc_perp                                   12131
## fsc_small                                  26208
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## chl_small                                   5672
## d1                                          1936
## d2                                            32
## fsc_perp                                       0
## fsc_small                                  10005
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## chl_small                                      3485
## d1                                             1488
## d2                                               32
## fsc_perp                                          0
## fsc_small                                     10056
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## chl_small                                   20725
## d1                                           1536
## d2                                             32
## fsc_perp                                     2160
## fsc_small                                   18821
## pe                                              0
## [1] "OOBobs pop.fctr.All.X...glmnet crypto: min < min of Train range: 1"
##       .rownames pop.fctr.All.X...glmnet chl_small    pe
## 21158     21158                  crypto     38267 30813
##                  id      cor.y exclude.as.feat cor.y.abs cor.high.X
## chl_small chl_small -0.1960482           FALSE 0.1960482    chl_big
## pe               pe  0.1065189           FALSE 0.1065189       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small  1.000000      37.81206   FALSE FALSE            FALSE
## pe         4.153846      20.68231   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## pe                      NA         7.564684e-71       FALSE      NA 58675
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small 3485               64784             64832             42016
## pe           0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## pe                                              0
## [1] "OOBobs pop.fctr.All.X...glmnet crypto: max > max of Train range: 5"
##       .rownames pop.fctr.All.X...glmnet chl_small    d1    pe
## 6263       6263                  crypto     64736 43200 58675
## 15306     15306                  crypto     64789 42816 56552
## 21158     21158                  crypto     38267 50896 30813
## 29132     29132                  crypto     64757 50528 53640
## 63492     63492                  crypto     64739 36496 58627
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small  1.000000     37.812059   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d1                      NA         1.011239e-40       FALSE      NA 54048
## pe                      NA         7.564684e-71       FALSE      NA 58675
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small 3485               64784             64832             42016
## d1        1328               48960             53456             41456
## pe           0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## d1                       51712              45904                5472
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## d1                     1600              1328                 1600
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## d1                      1616                              50896
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## d1                                   54048
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## d1                                   41808
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## d1                                      53040
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## d1                                    43904
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## d1                                     36496
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## d1                                    1792
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## d1                                    1936
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## d1                                       1488
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## d1                                     1536
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## d1                                           50896
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## d1                                         54048
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## d1                                         41808
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## d1                                            53040
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## d1                                          43904
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## d1                                           36496
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## d1                                          1792
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## d1                                          1936
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## d1                                             1488
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## d1                                           1536
## pe                                              0
## [1] "OOBobs total range outliers: 173"
## [1] "newobs pop.fctr.Final.All.X...glmnet ultra: min < min of Train range: 6"
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id    d1
## 2             2                         ultra     2       4 12960
## 10455     10455                         ultra 10455       0 18016
## 10456     10456                         ultra 10456       4 20640
## 37482     37482                         ultra 37482       1 16608
## 52853     52853                         ultra 52853   10609  1536
## 60920     60920                         ultra 60920       3  5648
##              id      cor.y exclude.as.feat  cor.y.abs cor.high.X freqRatio
## .pos       .pos 0.04197727           FALSE 0.04197727       time  1.000000
## cell_id cell_id 0.00203730           FALSE 0.00203730       <NA>  1.000000
## d1           d1 0.06264683           FALSE 0.06264683       <NA>  1.079365
##         percentUnique zeroVar   nzv is.cor.y.abs.low interaction.feat
## .pos       100.000000   FALSE FALSE            FALSE               NA
## cell_id     61.919770   FALSE FALSE            FALSE               NA
## d1           6.673855   FALSE FALSE            FALSE               NA
##         shapiro.test.p.value rsp_var_raw rsp_var   max  min
## .pos            4.856422e-37       FALSE      NA 72343    1
## cell_id         2.576068e-35       FALSE      NA 32081    0
## d1              1.011239e-40       FALSE      NA 54048 1328
##         max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos                  71551             72319             72328
## cell_id               30290             32032             32043
## d1                    48960             53456             41456
##         max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                   72341              72335                  76
## cell_id                32035              32078                 112
## d1                     51712              45904                5472
##         min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                   48                 1                    5
## cell_id                 2                 0                    2
## d1                   1600              1328                 1600
##         min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                     4                              63492
## cell_id                  5                              17616
## d1                    1616                              50896
##         max.pop.fctr.All.X...glmnet.nano max.pop.fctr.All.X...glmnet.pico
## .pos                               72337                            72316
## cell_id                            32076                            31980
## d1                                 54048                            41808
##         max.pop.fctr.All.X...glmnet.synecho
## .pos                                  72343
## cell_id                               32053
## d1                                    53040
##         max.pop.fctr.All.X...glmnet.ultra
## .pos                                72342
## cell_id                             32081
## d1                                  43904
##         min.pop.fctr.All.X...glmnet.crypto
## .pos                                  6263
## cell_id                               2802
## d1                                   36496
##         min.pop.fctr.All.X...glmnet.nano min.pop.fctr.All.X...glmnet.pico
## .pos                                  15                                3
## cell_id                                1                                6
## d1                                  1792                             1936
##         min.pop.fctr.All.X...glmnet.synecho
## .pos                                     42
## cell_id                                   7
## d1                                     1488
##         min.pop.fctr.All.X...glmnet.ultra
## .pos                                    2
## cell_id                                 0
## d1                                   1536
##         max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                       63492
## cell_id                                    17616
## d1                                         50896
##         max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                     72337
## cell_id                                  32076
## d1                                       54048
##         max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                     72316
## cell_id                                  31980
## d1                                       41808
##         max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                        72343
## cell_id                                     32053
## d1                                          53040
##         max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                      72342
## cell_id                                   32081
## d1                                        43904
##         min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                        6263
## cell_id                                     2802
## d1                                         36496
##         min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                        15
## cell_id                                      1
## d1                                        1792
##         min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                         3
## cell_id                                      6
## d1                                        1936
##         min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                           42
## cell_id                                         7
## d1                                           1488
##         min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                          2
## cell_id                                       0
## d1                                         1536
## [1] "newobs pop.fctr.Final.All.X...glmnet ultra: max > max of Train range: 8"
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id chl_small
## 8233       8233                         ultra  8233   23305     25133
## 23483     23483                         ultra 23483   11572     49184
## 23543     23543                         ultra 23543   11752     26531
## 52529     52529                         ultra 52529    9698     48869
## 60919     60919                         ultra 60919   32081     33403
## 66618     66618                         ultra 66618   15539     25595
## 70887     70887                         ultra 70887   27173     50160
## 72342     72342                         ultra 72342   31315     29435
##          d2 fsc_perp
## 8233  35952    51483
## 23483 22224    17824
## 23543 46016    37005
## 52529 27424    14291
## 60919 19888    15885
## 66618 35824    52669
## 70887 19120    13416
## 72342 18912    18301
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d2               d2  0.04550252           FALSE 0.04550252         d1
## fsc_perp   fsc_perp -0.12774069           FALSE 0.12774069       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos        1.00000    100.000000   FALSE FALSE            FALSE
## cell_id     1.00000     61.919770   FALSE FALSE            FALSE
## chl_small   1.00000     37.812059   FALSE FALSE            FALSE
## d2          1.02500      7.143845   FALSE FALSE            FALSE
## fsc_perp   21.58824     29.247187   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d2                      NA         8.244239e-28       FALSE      NA 54688
## fsc_perp                NA         2.470795e-26       FALSE      NA 63456
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos         1               71551             72319             72328
## cell_id      0               30290             32032             32043
## chl_small 3485               64784             64832             42016
## d2          32               54688             50960             43248
## fsc_perp     0               62173             55104             46880
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## chl_small                43307              47888               48888
## d2                       45792              45904                1712
## fsc_perp                 49589              51237                8528
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## chl_small             30251             12003                 3547
## d2                       48                32                   32
## fsc_perp               4563                 0                    0
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## chl_small              14485                              64789
## d2                        32                              49232
## fsc_perp                   0                              56579
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## chl_small                            64805
## d2                                   54096
## fsc_perp                             63456
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## chl_small                            40797
## d2                                   42064
## fsc_perp                             45952
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## chl_small                               64776
## d2                                      51872
## fsc_perp                                53571
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## chl_small                             50160
## d2                                    46016
## fsc_perp                              52669
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## chl_small                              38267
## d2                                     38048
## fsc_perp                               44157
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## chl_small                            27245
## d2                                      32
## fsc_perp                             12131
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## chl_small                             5672
## d2                                      32
## fsc_perp                                 0
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## chl_small                                3485
## d2                                         32
## fsc_perp                                    0
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## chl_small                             20725
## d2                                       32
## fsc_perp                               2160
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## chl_small                                    64789
## d2                                           49232
## fsc_perp                                     56579
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## chl_small                                  64805
## d2                                         54096
## fsc_perp                                   63456
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## chl_small                                  40797
## d2                                         42064
## fsc_perp                                   45952
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## chl_small                                     64776
## d2                                            51872
## fsc_perp                                      53571
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## chl_small                                   50160
## d2                                          46016
## fsc_perp                                    52669
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## chl_small                                    38267
## d2                                           38048
## fsc_perp                                     44157
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## chl_small                                  27245
## d2                                            32
## fsc_perp                                   12131
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## chl_small                                   5672
## d2                                            32
## fsc_perp                                       0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## chl_small                                      3485
## d2                                               32
## fsc_perp                                          0
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## chl_small                                   20725
## d2                                             32
## fsc_perp                                     2160
## [1] "newobs pop.fctr.Final.All.X...glmnet pico: min < min of Train range: 26"
##      .rownames pop.fctr.Final.All.X...glmnet chl_small fsc_small
## 396        396                          pico      5968     10232
## 824        824                          pico      7653     13984
## 1330      1330                          pico      6232     16581
## 1696      1696                          pico      6741     11712
## 2539      2539                          pico      6416     17803
## 2702      2702                          pico      5672     17885
##       .rownames pop.fctr.Final.All.X...glmnet chl_small fsc_small
## 2539       2539                          pico      6416     17803
## 2702       2702                          pico      5672     17885
## 12483     12483                          pico      6152     12056
## 19669     19669                          pico      6808     21800
## 23242     23242                          pico     11827     33133
## 60756     60756                          pico     12000     15115
##       .rownames pop.fctr.Final.All.X...glmnet chl_small fsc_small
## 23242     23242                          pico     11827     33133
## 27339     27339                          pico      9003     19853
## 27955     27955                          pico     10936     16475
## 31863     31863                          pico     14312     10005
## 32330     32330                          pico     11965     21280
## 60756     60756                          pico     12000     15115
##                  id      cor.y exclude.as.feat cor.y.abs cor.high.X
## chl_small chl_small -0.1960482           FALSE 0.1960482    chl_big
## fsc_small fsc_small -0.1175350           FALSE 0.1175350   fsc_perp
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small         1      37.81206   FALSE FALSE            FALSE
## fsc_small         1      30.47745   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small  3485               64784             64832             42016
## fsc_small 10005               65424             65365             53568
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## fsc_small                63547              56896               29088
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## fsc_small             24843             10011                10061
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## fsc_small              10376                              65405
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## fsc_small                            65424
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## fsc_small                            49208
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## fsc_small                               65416
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## fsc_small                             54885
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## fsc_small                              58219
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## fsc_small                            26208
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## fsc_small                            10005
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## fsc_small                               10056
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## fsc_small                             18821
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## fsc_small                                    65405
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## fsc_small                                  65424
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## fsc_small                                  49208
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## fsc_small                                     65416
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## fsc_small                                   54885
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## fsc_small                                    58219
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## fsc_small                                  26208
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## fsc_small                                  10005
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## fsc_small                                     10056
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## fsc_small                                   18821
## [1] "newobs pop.fctr.Final.All.X...glmnet pico: max > max of Train range: 4"
##       .rownames pop.fctr.Final.All.X...glmnet    d1 fsc_big
## 18348     18348                          pico 28544   32448
## 20094     20094                          pico 41808   32400
## 26156     26156                          pico  9264   32448
## 49143     49143                          pico 27728   32448
##              id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## d1           d1  0.06264683           FALSE 0.06264683       <NA>
## fsc_big fsc_big -0.03147761           FALSE 0.03147761       <NA>
##         freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## d1       1.079365    6.67385475   FALSE FALSE            FALSE
## fsc_big  2.153882    0.01658787   FALSE FALSE            FALSE
##         interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## d1                    NA         1.011239e-40       FALSE      NA 54048
## fsc_big               NA         3.294647e-68       FALSE      NA 32464
##           min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## d1       1328               48960             53456             41456
## fsc_big 32384               32448             32464             32432
##         max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## d1                     51712              45904                5472
## fsc_big                32464              32448               32416
##         min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## d1                   1600              1328                 1600
## fsc_big             32384             32384                32384
##         min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## d1                    1616                              50896
## fsc_big              32384                              32448
##         max.pop.fctr.All.X...glmnet.nano max.pop.fctr.All.X...glmnet.pico
## d1                                 54048                            41808
## fsc_big                            32448                            32448
##         max.pop.fctr.All.X...glmnet.synecho
## d1                                    53040
## fsc_big                               32448
##         max.pop.fctr.All.X...glmnet.ultra
## d1                                  43904
## fsc_big                             32448
##         min.pop.fctr.All.X...glmnet.crypto
## d1                                   36496
## fsc_big                              32432
##         min.pop.fctr.All.X...glmnet.nano min.pop.fctr.All.X...glmnet.pico
## d1                                  1792                             1936
## fsc_big                            32384                            32384
##         min.pop.fctr.All.X...glmnet.synecho
## d1                                     1488
## fsc_big                               32384
##         min.pop.fctr.All.X...glmnet.ultra
## d1                                   1536
## fsc_big                             32384
##         max.pop.fctr.Final.All.X...glmnet.crypto
## d1                                         50896
## fsc_big                                    32448
##         max.pop.fctr.Final.All.X...glmnet.nano
## d1                                       54048
## fsc_big                                  32448
##         max.pop.fctr.Final.All.X...glmnet.pico
## d1                                       41808
## fsc_big                                  32448
##         max.pop.fctr.Final.All.X...glmnet.synecho
## d1                                          53040
## fsc_big                                     32448
##         max.pop.fctr.Final.All.X...glmnet.ultra
## d1                                        43904
## fsc_big                                   32448
##         min.pop.fctr.Final.All.X...glmnet.crypto
## d1                                         36496
## fsc_big                                    32432
##         min.pop.fctr.Final.All.X...glmnet.nano
## d1                                        1792
## fsc_big                                  32384
##         min.pop.fctr.Final.All.X...glmnet.pico
## d1                                        1936
## fsc_big                                  32384
##         min.pop.fctr.Final.All.X...glmnet.synecho
## d1                                           1488
## fsc_big                                     32384
##         min.pop.fctr.Final.All.X...glmnet.ultra
## d1                                         1536
## fsc_big                                   32384
## [1] "newobs pop.fctr.Final.All.X...glmnet nano: min < min of Train range: 9"
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id chl_small
## 15           15                          nano    15      37     40597
## 22           22                          nano    22      59     45275
## 29           29                          nano    29      82     42499
## 34           34                          nano    34      96     45800
## 46           46                          nano    46     133     42077
## 11405     11405                          nano 11405    2903     42696
## 19890     19890                          nano 19890       1     44509
## 59277     59277                          nano 59277   27708     30013
## 70195     70195                          nano 70195   25242     27245
##          d2 time
## 15    28288   12
## 22     7520   12
## 29     6016   12
## 34    24208   12
## 46     8432   13
## 11405    32  111
## 19890 31984  192
## 59277 44512  541
## 70195 20560  624
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d2               d2  0.04550252           FALSE 0.04550252         d1
## time           time  0.04275996           FALSE 0.04275996    file_id
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos          1.000    100.000000   FALSE FALSE            FALSE
## cell_id       1.000     61.919770   FALSE FALSE            FALSE
## chl_small     1.000     37.812059   FALSE FALSE            FALSE
## d2            1.025      7.143845   FALSE FALSE            FALSE
## time          1.040      1.747256   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d2                      NA         8.244239e-28       FALSE      NA 54688
## time                    NA         1.186295e-39       FALSE      NA   643
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos         1               71551             72319             72328
## cell_id      0               30290             32032             32043
## chl_small 3485               64784             64832             42016
## d2          32               54688             50960             43248
## time        12                 636               642               642
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## chl_small                43307              47888               48888
## d2                       45792              45904                1712
## time                       643                643                  13
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## chl_small             30251             12003                 3547
## d2                       48                32                   32
## time                     13                12                   12
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## chl_small              14485                              64789
## d2                        32                              49232
## time                      12                                572
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## chl_small                            64805
## d2                                   54096
## time                                   643
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## chl_small                            40797
## d2                                   42064
## time                                   642
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## chl_small                               64776
## d2                                      51872
## time                                      643
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## chl_small                             50160
## d2                                    46016
## time                                    643
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## chl_small                              38267
## d2                                     38048
## time                                      66
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## chl_small                            27245
## d2                                      32
## time                                    12
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## chl_small                             5672
## d2                                      32
## time                                    12
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## chl_small                                3485
## d2                                         32
## time                                       12
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## chl_small                             20725
## d2                                       32
## time                                     12
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## chl_small                                    64789
## d2                                           49232
## time                                           572
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## chl_small                                  64805
## d2                                         54096
## time                                         643
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## chl_small                                  40797
## d2                                         42064
## time                                         642
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## chl_small                                     64776
## d2                                            51872
## time                                            643
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## chl_small                                   50160
## d2                                          46016
## time                                          643
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## chl_small                                    38267
## d2                                           38048
## time                                            66
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## chl_small                                  27245
## d2                                            32
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## chl_small                                   5672
## d2                                            32
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## chl_small                                      3485
## d2                                               32
## time                                             12
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## chl_small                                   20725
## d2                                             32
## time                                           12
## [1] "newobs pop.fctr.Final.All.X...glmnet nano: max > max of Train range: 19"
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id    d1    d2
## 876         876                          nano   876    2321 47824 53056
## 1718       1718                          nano  1718    4606 41520 46208
## 2887       2887                          nano  2887    7880 39904 44032
## 17371     17371                          nano 17371   20700 45648 54096
## 30453     30453                          nano 30453    7125 54048 45056
## 43815     43815                          nano 43815   17740 45456 50640
## 44731     44731                          nano 44731   20093 31040 40080
## 51111     51111                          nano 51111    6102 50320 47584
## 51444     51444                          nano 51444    6906 40496 51856
## 57830     57830                          nano 57830   23840 40384 51632
## 59421     59421                          nano 59421   28081 43376 52192
## 60917     60917                          nano 60917   32076 35680 36848
## 64561     64561                          nano 64561    9802 47040 46800
## 65616     65616                          nano 65616   12670 42064 45808
## 70085     70085                          nano 70085   24927 45136 52784
## 70897     70897                          nano 70897   27194 40624 53136
## 72322     72322                          nano 72322   31258 10608 13328
## 72325     72325                          nano 72325   31263 17328 15040
## 72337     72337                          nano 72337   31303 12912 20448
##       fsc_perp fsc_small    pe time
## 876      63456     63640 20232   20
## 1718     49587     63147 50731   26
## 2887     53051     62832 53245   36
## 17371    46536     63317 16133  168
## 30453    45077     61456 17493  306
## 43815    51181     65424 14637  424
## 44731    53755     63171 54117  431
## 51111    60403     64619 21979  480
## 51444    47691     63971 20760  482
## 57830    42419     57699  8936  530
## 59421    43976     61837 13131  542
## 60917    37659     56301  3859  553
## 64561    58957     65397 24707  580
## 65616    60195     65373 45571  588
## 70085    53784     65341 25395  624
## 70897    46573     59419 13741  630
## 72322    33699     47848  2979  642
## 72325    29789     47653  1171  642
## 72337    27648     45987  2277  643
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## d2               d2  0.04550252           FALSE 0.04550252         d1
## fsc_perp   fsc_perp -0.12774069           FALSE 0.12774069       <NA>
## fsc_small fsc_small -0.11753499           FALSE 0.11753499   fsc_perp
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
## time           time  0.04275996           FALSE 0.04275996    file_id
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos       1.000000    100.000000   FALSE FALSE            FALSE
## cell_id    1.000000     61.919770   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## d2         1.025000      7.143845   FALSE FALSE            FALSE
## fsc_perp  21.588235     29.247187   FALSE FALSE            FALSE
## fsc_small  1.000000     30.477454   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
## time       1.040000      1.747256   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## d1                      NA         1.011239e-40       FALSE      NA 54048
## d2                      NA         8.244239e-28       FALSE      NA 54688
## fsc_perp                NA         2.470795e-26       FALSE      NA 63456
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
## pe                      NA         7.564684e-71       FALSE      NA 58675
## time                    NA         1.186295e-39       FALSE      NA   643
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos          1               71551             72319             72328
## cell_id       0               30290             32032             32043
## d1         1328               48960             53456             41456
## d2           32               54688             50960             43248
## fsc_perp      0               62173             55104             46880
## fsc_small 10005               65424             65365             53568
## pe            0               58267             50416             45107
## time         12                 636               642               642
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## d1                       51712              45904                5472
## d2                       45792              45904                1712
## fsc_perp                 49589              51237                8528
## fsc_small                63547              56896               29088
## pe                       43744              46688               31125
## time                       643                643                  13
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## d1                     1600              1328                 1600
## d2                       48                32                   32
## fsc_perp               4563                 0                    0
## fsc_small             24843             10011                10061
## pe                        0                 0                 5005
## time                     13                12                   12
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## d1                      1616                              50896
## d2                        32                              49232
## fsc_perp                   0                              56579
## fsc_small              10376                              65405
## pe                         0                              58675
## time                      12                                572
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## d1                                   54048
## d2                                   54096
## fsc_perp                             63456
## fsc_small                            65424
## pe                                   54117
## time                                   643
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## d1                                   41808
## d2                                   42064
## fsc_perp                             45952
## fsc_small                            49208
## pe                                    8091
## time                                   642
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## d1                                      53040
## d2                                      51872
## fsc_perp                                53571
## fsc_small                               65416
## pe                                      56280
## time                                      643
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## d1                                    43904
## d2                                    46016
## fsc_perp                              52669
## fsc_small                             54885
## pe                                    14963
## time                                    643
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## d1                                     36496
## d2                                     38048
## fsc_perp                               44157
## fsc_small                              58219
## pe                                     30813
## time                                      66
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## d1                                    1792
## d2                                      32
## fsc_perp                             12131
## fsc_small                            26208
## pe                                       0
## time                                    12
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## d1                                    1936
## d2                                      32
## fsc_perp                                 0
## fsc_small                            10005
## pe                                       0
## time                                    12
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## d1                                       1488
## d2                                         32
## fsc_perp                                    0
## fsc_small                               10056
## pe                                       2773
## time                                       12
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## d1                                     1536
## d2                                       32
## fsc_perp                               2160
## fsc_small                             18821
## pe                                        0
## time                                     12
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## d1                                           50896
## d2                                           49232
## fsc_perp                                     56579
## fsc_small                                    65405
## pe                                           58675
## time                                           572
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## d1                                         54048
## d2                                         54096
## fsc_perp                                   63456
## fsc_small                                  65424
## pe                                         54117
## time                                         643
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## d1                                         41808
## d2                                         42064
## fsc_perp                                   45952
## fsc_small                                  49208
## pe                                          8091
## time                                         642
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## d1                                            53040
## d2                                            51872
## fsc_perp                                      53571
## fsc_small                                     65416
## pe                                            56280
## time                                            643
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## d1                                          43904
## d2                                          46016
## fsc_perp                                    52669
## fsc_small                                   54885
## pe                                          14963
## time                                          643
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## d1                                           36496
## d2                                           38048
## fsc_perp                                     44157
## fsc_small                                    58219
## pe                                           30813
## time                                            66
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## d1                                          1792
## d2                                            32
## fsc_perp                                   12131
## fsc_small                                  26208
## pe                                             0
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## d1                                          1936
## d2                                            32
## fsc_perp                                       0
## fsc_small                                  10005
## pe                                             0
## time                                          12
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## d1                                             1488
## d2                                               32
## fsc_perp                                          0
## fsc_small                                     10056
## pe                                             2773
## time                                             12
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## d1                                           1536
## d2                                             32
## fsc_perp                                     2160
## fsc_small                                   18821
## pe                                              0
## time                                           12
## [1] "newobs pop.fctr.Final.All.X...glmnet synecho: min < min of Train range: 68"
##       .rownames pop.fctr.Final.All.X...glmnet chl_small    d1 fsc_small
## 1304       1304                       synecho      3675 14256     20483
## 1614       1614                       synecho      3485  4224     15827
## 3006       3006                       synecho     12989  6848     24248
## 13772     13772                       synecho     18429  5696     37984
## 17188     17188                       synecho     17067  8736     35368
## 23562     23562                       synecho     16157  3936     26157
##         pe
## 1304  5003
## 1614  5168
## 3006  4251
## 13772 4624
## 17188 4797
## 23562 4949
##       .rownames pop.fctr.Final.All.X...glmnet chl_small    d1 fsc_small
## 29734     29734                       synecho     19488 30656     45640
## 42084     42084                       synecho     13776  5344     35333
## 42985     42985                       synecho     16011  7312     30752
## 46052     46052                       synecho     17525  9680     33555
## 57484     57484                       synecho     12851 19056     29501
## 70211     70211                       synecho     12667 20800     33069
##         pe
## 29734 4405
## 42084 3467
## 42985 4864
## 46052 4883
## 57484 4827
## 70211 4357
##       .rownames pop.fctr.Final.All.X...glmnet chl_small    d1 fsc_small
## 69562     69562                       synecho     12896  8064     27181
## 70211     70211                       synecho     12667 20800     33069
## 71096     71096                       synecho     17800 22368     33277
## 71353     71353                       synecho     14675 15552     29461
## 71638     71638                       synecho     15384 17984     31781
## 71996     71996                       synecho     14005 25328     34144
##         pe
## 69562 4504
## 70211 4357
## 71096 4952
## 71353 4544
## 71638 4819
## 71996 4453
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## fsc_small fsc_small -0.11753499           FALSE 0.11753499   fsc_perp
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small  1.000000     37.812059   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## fsc_small  1.000000     30.477454   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d1                      NA         1.011239e-40       FALSE      NA 54048
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
## pe                      NA         7.564684e-71       FALSE      NA 58675
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small  3485               64784             64832             42016
## d1         1328               48960             53456             41456
## fsc_small 10005               65424             65365             53568
## pe            0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## d1                       51712              45904                5472
## fsc_small                63547              56896               29088
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## d1                     1600              1328                 1600
## fsc_small             24843             10011                10061
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## d1                      1616                              50896
## fsc_small              10376                              65405
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## d1                                   54048
## fsc_small                            65424
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## d1                                   41808
## fsc_small                            49208
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## d1                                      53040
## fsc_small                               65416
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## d1                                    43904
## fsc_small                             54885
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## d1                                     36496
## fsc_small                              58219
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## d1                                    1792
## fsc_small                            26208
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## d1                                    1936
## fsc_small                            10005
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## d1                                       1488
## fsc_small                               10056
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## d1                                     1536
## fsc_small                             18821
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## d1                                           50896
## fsc_small                                    65405
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## d1                                         54048
## fsc_small                                  65424
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## d1                                         41808
## fsc_small                                  49208
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## d1                                            53040
## fsc_small                                     65416
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## d1                                          43904
## fsc_small                                   54885
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## d1                                           36496
## fsc_small                                    58219
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## d1                                          1792
## fsc_small                                  26208
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## d1                                          1936
## fsc_small                                  10005
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## d1                                             1488
## fsc_small                                     10056
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## d1                                           1536
## fsc_small                                   18821
## pe                                              0
## [1] "newobs pop.fctr.Final.All.X...glmnet synecho: max > max of Train range: 28"
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id chl_small
## 3519       3519                       synecho  3519    9643     44493
## 5319       5319                       synecho  5319   14799     35125
## 7991       7991                       synecho  7991   22556     54011
## 9689       9689                       synecho  9689   27526     64776
## 13776     13776                       synecho 13776    9824     43357
## 23448     23448                       synecho 23448   11458     47072
##          d1    d2 fsc_perp fsc_small    pe
## 3519  14160  7840     1035     17752 40971
## 5319  37968 40400    38208     52387 50397
## 7991   5520 14512    14056     29315 49925
## 9689   7744 16160    27416     46179 55357
## 13776  8032  4688    23056     38944 46443
## 23448  7824  4672     8157     26499 38419
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id chl_small
## 31396     31396                       synecho 31396   10157     36755
## 48960     48960                       synecho 48960     124     57085
## 58296     58296                       synecho 58296   25066     45557
## 60290     60290                       synecho 60290   30407     45064
## 62460     62460                       synecho 62460    4118     47125
## 67647     67647                       synecho 67647   18280     44603
##          d1    d2 fsc_perp fsc_small    pe
## 31396 47568 51872    52683     65349 44616
## 48960  4336  1040    11843     29632 44213
## 58296 12000 22032     3003     22616 41003
## 60290  4400  7328     6704     24840 39651
## 62460  4320  2304     4037     18229 40875
## 67647 37200 47376    49253     63763 47880
##       .rownames pop.fctr.Final.All.X...glmnet  .pos cell_id chl_small
## 68935     68935                       synecho 68935   21773     49792
## 69447     69447                       synecho 69447   23100     44331
## 69623     69623                       synecho 69623   23581     60560
## 71612     71612                       synecho 71612   29172     47803
## 72034     72034                       synecho 72034   30430     36096
## 72343     72343                       synecho 72343   31316     21288
##          d1    d2 fsc_perp fsc_small    pe
## 68935  6304   944    12099     32053 47595
## 69447  3264   448     1851     18643 25437
## 69623 28624 28928    26795     46453 50059
## 71612 42320 48976    53571     65416 50821
## 72034 53040 48400    47851     62496 43957
## 72343  6592  4704    22192     37096 17995
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## .pos           .pos  0.04197727           FALSE 0.04197727       time
## cell_id     cell_id  0.00203730           FALSE 0.00203730       <NA>
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## d2               d2  0.04550252           FALSE 0.04550252         d1
## fsc_perp   fsc_perp -0.12774069           FALSE 0.12774069       <NA>
## fsc_small fsc_small -0.11753499           FALSE 0.11753499   fsc_perp
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## .pos       1.000000    100.000000   FALSE FALSE            FALSE
## cell_id    1.000000     61.919770   FALSE FALSE            FALSE
## chl_small  1.000000     37.812059   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## d2         1.025000      7.143845   FALSE FALSE            FALSE
## fsc_perp  21.588235     29.247187   FALSE FALSE            FALSE
## fsc_small  1.000000     30.477454   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## .pos                    NA         4.856422e-37       FALSE      NA 72343
## cell_id                 NA         2.576068e-35       FALSE      NA 32081
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d1                      NA         1.011239e-40       FALSE      NA 54048
## d2                      NA         8.244239e-28       FALSE      NA 54688
## fsc_perp                NA         2.470795e-26       FALSE      NA 63456
## fsc_small               NA         3.816202e-31       FALSE      NA 65424
## pe                      NA         7.564684e-71       FALSE      NA 58675
##             min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## .pos          1               71551             72319             72328
## cell_id       0               30290             32032             32043
## chl_small  3485               64784             64832             42016
## d1         1328               48960             53456             41456
## d2           32               54688             50960             43248
## fsc_perp      0               62173             55104             46880
## fsc_small 10005               65424             65365             53568
## pe            0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## .pos                     72341              72335                  76
## cell_id                  32035              32078                 112
## chl_small                43307              47888               48888
## d1                       51712              45904                5472
## d2                       45792              45904                1712
## fsc_perp                 49589              51237                8528
## fsc_small                63547              56896               29088
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## .pos                     48                 1                    5
## cell_id                   2                 0                    2
## chl_small             30251             12003                 3547
## d1                     1600              1328                 1600
## d2                       48                32                   32
## fsc_perp               4563                 0                    0
## fsc_small             24843             10011                10061
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## .pos                       4                              63492
## cell_id                    5                              17616
## chl_small              14485                              64789
## d1                      1616                              50896
## d2                        32                              49232
## fsc_perp                   0                              56579
## fsc_small              10376                              65405
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## .pos                                 72337
## cell_id                              32076
## chl_small                            64805
## d1                                   54048
## d2                                   54096
## fsc_perp                             63456
## fsc_small                            65424
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## .pos                                 72316
## cell_id                              31980
## chl_small                            40797
## d1                                   41808
## d2                                   42064
## fsc_perp                             45952
## fsc_small                            49208
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## .pos                                    72343
## cell_id                                 32053
## chl_small                               64776
## d1                                      53040
## d2                                      51872
## fsc_perp                                53571
## fsc_small                               65416
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## .pos                                  72342
## cell_id                               32081
## chl_small                             50160
## d1                                    43904
## d2                                    46016
## fsc_perp                              52669
## fsc_small                             54885
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## .pos                                    6263
## cell_id                                 2802
## chl_small                              38267
## d1                                     36496
## d2                                     38048
## fsc_perp                               44157
## fsc_small                              58219
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## .pos                                    15
## cell_id                                  1
## chl_small                            27245
## d1                                    1792
## d2                                      32
## fsc_perp                             12131
## fsc_small                            26208
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## .pos                                     3
## cell_id                                  6
## chl_small                             5672
## d1                                    1936
## d2                                      32
## fsc_perp                                 0
## fsc_small                            10005
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## .pos                                       42
## cell_id                                     7
## chl_small                                3485
## d1                                       1488
## d2                                         32
## fsc_perp                                    0
## fsc_small                               10056
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## .pos                                      2
## cell_id                                   0
## chl_small                             20725
## d1                                     1536
## d2                                       32
## fsc_perp                               2160
## fsc_small                             18821
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                         63492
## cell_id                                      17616
## chl_small                                    64789
## d1                                           50896
## d2                                           49232
## fsc_perp                                     56579
## fsc_small                                    65405
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## .pos                                       72337
## cell_id                                    32076
## chl_small                                  64805
## d1                                         54048
## d2                                         54096
## fsc_perp                                   63456
## fsc_small                                  65424
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## .pos                                       72316
## cell_id                                    31980
## chl_small                                  40797
## d1                                         41808
## d2                                         42064
## fsc_perp                                   45952
## fsc_small                                  49208
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                          72343
## cell_id                                       32053
## chl_small                                     64776
## d1                                            53040
## d2                                            51872
## fsc_perp                                      53571
## fsc_small                                     65416
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                        72342
## cell_id                                     32081
## chl_small                                   50160
## d1                                          43904
## d2                                          46016
## fsc_perp                                    52669
## fsc_small                                   54885
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## .pos                                          6263
## cell_id                                       2802
## chl_small                                    38267
## d1                                           36496
## d2                                           38048
## fsc_perp                                     44157
## fsc_small                                    58219
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## .pos                                          15
## cell_id                                        1
## chl_small                                  27245
## d1                                          1792
## d2                                            32
## fsc_perp                                   12131
## fsc_small                                  26208
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## .pos                                           3
## cell_id                                        6
## chl_small                                   5672
## d1                                          1936
## d2                                            32
## fsc_perp                                       0
## fsc_small                                  10005
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## .pos                                             42
## cell_id                                           7
## chl_small                                      3485
## d1                                             1488
## d2                                               32
## fsc_perp                                          0
## fsc_small                                     10056
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## .pos                                            2
## cell_id                                         0
## chl_small                                   20725
## d1                                           1536
## d2                                             32
## fsc_perp                                     2160
## fsc_small                                   18821
## pe                                              0
## [1] "newobs pop.fctr.Final.All.X...glmnet crypto: min < min of Train range: 1"
##       .rownames pop.fctr.Final.All.X...glmnet chl_small    pe
## 21158     21158                        crypto     38267 30813
##                  id      cor.y exclude.as.feat cor.y.abs cor.high.X
## chl_small chl_small -0.1960482           FALSE 0.1960482    chl_big
## pe               pe  0.1065189           FALSE 0.1065189       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small  1.000000      37.81206   FALSE FALSE            FALSE
## pe         4.153846      20.68231   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## pe                      NA         7.564684e-71       FALSE      NA 58675
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small 3485               64784             64832             42016
## pe           0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## pe                                              0
## [1] "newobs pop.fctr.Final.All.X...glmnet crypto: max > max of Train range: 5"
##       .rownames pop.fctr.Final.All.X...glmnet chl_small    d1    pe
## 6263       6263                        crypto     64736 43200 58675
## 15306     15306                        crypto     64789 42816 56552
## 21158     21158                        crypto     38267 50896 30813
## 29132     29132                        crypto     64757 50528 53640
## 63492     63492                        crypto     64739 36496 58627
##                  id       cor.y exclude.as.feat  cor.y.abs cor.high.X
## chl_small chl_small -0.19604819           FALSE 0.19604819    chl_big
## d1               d1  0.06264683           FALSE 0.06264683       <NA>
## pe               pe  0.10651893           FALSE 0.10651893       <NA>
##           freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## chl_small  1.000000     37.812059   FALSE FALSE            FALSE
## d1         1.079365      6.673855   FALSE FALSE            FALSE
## pe         4.153846     20.682315   FALSE FALSE            FALSE
##           interaction.feat shapiro.test.p.value rsp_var_raw rsp_var   max
## chl_small               NA         5.340544e-19       FALSE      NA 64832
## d1                      NA         1.011239e-40       FALSE      NA 54048
## pe                      NA         7.564684e-71       FALSE      NA 58675
##            min max.pop.fctr.crypto max.pop.fctr.nano max.pop.fctr.pico
## chl_small 3485               64784             64832             42016
## d1        1328               48960             53456             41456
## pe           0               58267             50416             45107
##           max.pop.fctr.synecho max.pop.fctr.ultra min.pop.fctr.crypto
## chl_small                43307              47888               48888
## d1                       51712              45904                5472
## pe                       43744              46688               31125
##           min.pop.fctr.nano min.pop.fctr.pico min.pop.fctr.synecho
## chl_small             30251             12003                 3547
## d1                     1600              1328                 1600
## pe                        0                 0                 5005
##           min.pop.fctr.ultra max.pop.fctr.All.X...glmnet.crypto
## chl_small              14485                              64789
## d1                      1616                              50896
## pe                         0                              58675
##           max.pop.fctr.All.X...glmnet.nano
## chl_small                            64805
## d1                                   54048
## pe                                   54117
##           max.pop.fctr.All.X...glmnet.pico
## chl_small                            40797
## d1                                   41808
## pe                                    8091
##           max.pop.fctr.All.X...glmnet.synecho
## chl_small                               64776
## d1                                      53040
## pe                                      56280
##           max.pop.fctr.All.X...glmnet.ultra
## chl_small                             50160
## d1                                    43904
## pe                                    14963
##           min.pop.fctr.All.X...glmnet.crypto
## chl_small                              38267
## d1                                     36496
## pe                                     30813
##           min.pop.fctr.All.X...glmnet.nano
## chl_small                            27245
## d1                                    1792
## pe                                       0
##           min.pop.fctr.All.X...glmnet.pico
## chl_small                             5672
## d1                                    1936
## pe                                       0
##           min.pop.fctr.All.X...glmnet.synecho
## chl_small                                3485
## d1                                       1488
## pe                                       2773
##           min.pop.fctr.All.X...glmnet.ultra
## chl_small                             20725
## d1                                     1536
## pe                                        0
##           max.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    64789
## d1                                           50896
## pe                                           58675
##           max.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  64805
## d1                                         54048
## pe                                         54117
##           max.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                  40797
## d1                                         41808
## pe                                          8091
##           max.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                     64776
## d1                                            53040
## pe                                            56280
##           max.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   50160
## d1                                          43904
## pe                                          14963
##           min.pop.fctr.Final.All.X...glmnet.crypto
## chl_small                                    38267
## d1                                           36496
## pe                                           30813
##           min.pop.fctr.Final.All.X...glmnet.nano
## chl_small                                  27245
## d1                                          1792
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.pico
## chl_small                                   5672
## d1                                          1936
## pe                                             0
##           min.pop.fctr.Final.All.X...glmnet.synecho
## chl_small                                      3485
## d1                                             1488
## pe                                             2773
##           min.pop.fctr.Final.All.X...glmnet.ultra
## chl_small                                   20725
## d1                                           1536
## pe                                              0
## [1] "newobs total range outliers: 173"
```

```
## [1] "glb_sel_mdl_id: All.X###glmnet"
```

```
## [1] "glb_fin_mdl_id: Final.All.X###glmnet"
```

```
## [1] "Cross Validation issues:"
##        MFO###myMFO_classfr  Random###myrandom_classfr 
##                          0                          0 
## Max.cor.Y.rcv.1X1###glmnet             All.X###glmnet 
##                          0                          0 
##              All.X###rpart                 All.X###rf 
##                          0                          0 
##       Final.All.X###glmnet 
##                          0
```

```
##                                 max.Accuracy.OOB max.Kappa.OOB
## All.X###rf                             0.9349497   0.912397767
## All.X###glmnet                         0.8552195   0.804899106
## Interact.High.cor.Y##rcv#glmnet        0.7521840   0.665566956
## Max.cor.Y##rcv#rpart                   0.7124295   0.611994574
## Max.cor.Y.rcv.1X1###glmnet             0.7062092   0.601619571
## MFO###myMFO_classfr                    0.2883446   0.000000000
## All.X###rpart                          0.2883446   0.000000000
## Random###myrandom_classfr              0.2550315  -0.003318276
```

```
## [1] "All.X###glmnet OOB confusion matrix & accuracy: "
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       5   33    0      13     0
##   nano         0 5311    1       7  1030
##   pico         0    1 8977      67  1385
##   synecho      1    6  141    8914    11
##   ultra        0  993 1543       5  7728
```

```
##        .freqRatio.Fit .freqRatio.OOB .freqRatio.Tst .n.Fit .n.New.crypto
## .dummy              1              1              1  36171             6
##        .n.New.nano .n.New.pico .n.New.synecho .n.New.ultra .n.OOB
## .dummy        6344       10662           9006        10154  36172
##        .n.Trn.crypto .n.Trn.nano .n.Trn.pico .n.Trn.synecho .n.Trn.ultra
## .dummy            51        6349       10430           9073        10268
##        .n.Tst .n.fit .n.new .n.trn err.abs.OOB.mean err.abs.fit.mean
## .dummy  36172  36171  36172  36171       0.05048317       0.04989861
##        err.abs.new.mean err.abs.trn.mean err.abs.OOB.sum err.abs.fit.sum
## .dummy       0.05048317       0.04989861        1826.077        1804.883
##        err.abs.new.sum err.abs.trn.sum
## .dummy        1826.077        1804.883
##   .freqRatio.Fit   .freqRatio.OOB   .freqRatio.Tst           .n.Fit 
##     1.000000e+00     1.000000e+00     1.000000e+00     3.617100e+04 
##    .n.New.crypto      .n.New.nano      .n.New.pico   .n.New.synecho 
##     6.000000e+00     6.344000e+03     1.066200e+04     9.006000e+03 
##     .n.New.ultra           .n.OOB    .n.Trn.crypto      .n.Trn.nano 
##     1.015400e+04     3.617200e+04     5.100000e+01     6.349000e+03 
##      .n.Trn.pico   .n.Trn.synecho     .n.Trn.ultra           .n.Tst 
##     1.043000e+04     9.073000e+03     1.026800e+04     3.617200e+04 
##           .n.fit           .n.new           .n.trn err.abs.OOB.mean 
##     3.617100e+04     3.617200e+04     3.617100e+04     5.048317e-02 
## err.abs.fit.mean err.abs.new.mean err.abs.trn.mean  err.abs.OOB.sum 
##     4.989861e-02     5.048317e-02     4.989861e-02     1.826077e+03 
##  err.abs.fit.sum  err.abs.new.sum  err.abs.trn.sum 
##     1.804883e+03     1.826077e+03     1.804883e+03
```

```
## [1] "Final.All.X###glmnet new confusion matrix & accuracy: "
##          Prediction
## Reference crypto nano pico synecho ultra
##   crypto       5   33    0      13     0
##   nano         0 5311    1       7  1030
##   pico         0    1 8977      67  1385
##   synecho      1    6  141    8914    11
##   ultra        0  993 1543       5  7728
```

```
##           All.X...glmnet.imp.crypto.x All.X...glmnet.imp.nano.x
## .pos                         71.25581                  71.25591
## .rnorm                       71.25581                  71.25581
## cell_id                      71.25581                  71.25773
## chl_small                    71.28856                  71.51482
## d1                           71.26464                  71.22409
## d2                           71.26731                  71.23331
## file_id                      71.25581                  98.63535
## fsc_big                     100.00000                  71.25581
## fsc_perp                     71.29304                  71.33496
## fsc_small                    71.28422                  71.38090
## pe                           71.39564                  71.17275
## time                         71.25581                  71.45262
##           All.X...glmnet.imp.pico.x All.X...glmnet.imp.synecho.x
## .pos                       71.25481                     71.25581
## .rnorm                     73.77245                     71.25581
## cell_id                    71.25581                     71.25581
## chl_small                  71.12234                     71.02547
## d1                         71.27978                     71.25581
## d2                         71.27942                     71.25581
## file_id                     0.00000                     71.25581
## fsc_big                    65.00687                     71.25581
## fsc_perp                   71.19554                     71.22825
## fsc_small                  71.14890                     71.18930
## pe                         70.99887                     71.57685
## time                       70.71917                     71.25581
##           All.X...glmnet.imp.ultra.x All.X___glmnet.imp
## .pos                        71.25581                 -1
## .rnorm                      71.25581                 -2
## cell_id                     71.25581                 -3
## chl_small                   71.31703                 -4
## d1                          71.25581                 -5
## d2                          71.25453                 -6
## file_id                     80.87732                 -7
## fsc_big                     68.22399                 -8
## fsc_perp                    71.24265                 -9
## fsc_small                   71.26054                -10
## pe                          71.15222                -11
## time                        71.25581                -12
##           All.X...glmnet.imp.crypto.y All.X...glmnet.imp.nano.y
## .pos                         71.25581                  71.25591
## .rnorm                       71.25581                  71.25581
## cell_id                      71.25581                  71.25773
## chl_small                    71.28856                  71.51482
## d1                           71.26464                  71.22409
## d2                           71.26731                  71.23331
## file_id                      71.25581                  98.63535
## fsc_big                     100.00000                  71.25581
## fsc_perp                     71.29304                  71.33496
## fsc_small                    71.28422                  71.38090
## pe                           71.39564                  71.17275
## time                         71.25581                  71.45262
##           All.X...glmnet.imp.pico.y All.X...glmnet.imp.synecho.y
## .pos                       71.25481                     71.25581
## .rnorm                     73.77245                     71.25581
## cell_id                    71.25581                     71.25581
## chl_small                  71.12234                     71.02547
## d1                         71.27978                     71.25581
## d2                         71.27942                     71.25581
## file_id                     0.00000                     71.25581
## fsc_big                    65.00687                     71.25581
## fsc_perp                   71.19554                     71.22825
## fsc_small                  71.14890                     71.18930
## pe                         70.99887                     71.57685
## time                       70.71917                     71.25581
##           All.X...glmnet.imp.ultra.y Final.All.X___glmnet.imp
## .pos                        71.25581                       -1
## .rnorm                      71.25581                       -2
## cell_id                     71.25581                       -3
## chl_small                   71.31703                       -4
## d1                          71.25581                       -5
## d2                          71.25453                       -6
## file_id                     80.87732                       -7
## fsc_big                     68.22399                       -8
## fsc_perp                    71.24265                       -9
## fsc_small                   71.26054                      -10
## pe                          71.15222                      -11
## time                        71.25581                      -12
```

```
## [1] "glbObsNew prediction stats:"
```

```
## 
##  crypto    nano    pico synecho   ultra 
##       6    6344   10662    9006   10154
```

```
##                   label step_major step_minor label_minor     bgn     end
## 16     predict.data.new          8          0           0 742.179 801.151
## 17 display.session.info          9          0           0 801.152      NA
##    elapsed
## 16  58.972
## 17      NA
```

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 
#```{r q1, cache=FALSE}
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
#```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                      label step_major step_minor label_minor     bgn
## 10              fit.models          6          0           0  68.735
## 11              fit.models          6          1           1 561.808
## 16        predict.data.new          8          0           0 742.179
## 12              fit.models          6          2           2 646.152
## 15       fit.data.training          7          1           1 710.661
## 13              fit.models          6          3           3 684.731
## 2             inspect.data          2          0           0  33.546
## 1              import.data          1          0           0  17.046
## 9          select.features          5          0           0  62.592
## 3               scrub.data          2          1           1  54.876
## 5         extract.features          3          0           0  58.401
## 6      manage.missing.data          3          1           1  59.911
## 8  partition.data.training          4          0           0  61.329
## 14       fit.data.training          7          0           0 710.188
## 7             cluster.data          3          2           2  61.204
## 4           transform.data          2          2           2  58.347
##        end elapsed duration
## 10 561.807 493.072  493.072
## 11 646.152  84.344   84.344
## 16 801.151  58.972   58.972
## 12 684.730  38.578   38.578
## 15 742.179  31.518   31.518
## 13 710.187  25.456   25.456
## 2   54.875  21.330   21.329
## 1   33.546  16.500   16.500
## 9   68.734   6.143    6.142
## 3   58.347   3.471    3.471
## 5   59.910   1.509    1.509
## 6   61.203   1.292    1.292
## 8   62.591   1.263    1.262
## 14 710.661   0.473    0.473
## 7   61.329   0.125    0.125
## 4   58.400   0.053    0.053
## [1] "Total Elapsed Time: 801.151 secs"
```

![](seaflow_base_files/figure-html/display.session.info-1.png) 

```
##                              label step_major step_minor      label_minor
## 5 fit.models_0_Interact.High.cor.Y          1          4           glmnet
## 4   fit.models_0_Max.cor.Y.rcv.*X*          1          3           glmnet
## 2                 fit.models_0_MFO          1          1    myMFO_classfr
## 3              fit.models_0_Random          1          2 myrandom_classfr
## 1                 fit.models_0_bgn          1          0            setup
## 6           fit.models_0_Low.cor.X          1          5           glmnet
##       bgn     end elapsed duration
## 5  90.043 561.782 471.740  471.739
## 4  71.711  90.043  18.332   18.332
## 2  69.417  70.716   1.299    1.299
## 3  70.717  71.710   0.993    0.993
## 1  69.375  69.416   0.041    0.041
## 6 561.783 561.797   0.015    0.014
## [1] "Total Elapsed Time: 561.797 secs"
```

![](seaflow_base_files/figure-html/display.session.info-2.png) 
