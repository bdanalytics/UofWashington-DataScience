# Kaggle Facial Points Detection: Univ of Montreal: left_eye_center_x regression:: Faces_tmplt
bdanalytics  

**  **    
**Date: (Tue) Jan 05, 2016**    

# Introduction:  

Data: 
Source: 
    Training:   https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip  
    New:        https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip  
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
glbCores <- 6 # of cores on machine - 2
registerDoMC(glbCores) 

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
#   url/name = "<pointer>"; if url specifies a zip file, name = "<filename>
#   sep = choose from c(NULL, "\t")
glbObsTrnFile <- list(url = "https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip",
                        name = "training/training.csv") 

glbObsNewFile <- list(url = "https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip",
                      name = "test/test.csv") # default OR
    #list(splitSpecs = list(method = NULL #select from c(NULL, "condition", "sample", "copy")
    #                      ,nRatio = 0.3 # > 0 && < 1 if method == "sample" 
    #                      ,seed = 123 # any integer or glbObsTrnPartitionSeed if method == "sample" 
    #                      ,condition = # or 'is.na(<var>)'; '<var> <condition_operator> <value>'    
    #                      )
    #    )                   

glbInpMerge <- NULL #: default
#     list(fnames = c("<fname1>", "<fname2>")) # files will be concatenated

glb_is_separate_newobs_dataset <- TRUE    # or TRUE
    glb_split_entity_newobs_datasets <- TRUE  # FALSE not supported - use "copy" for glbObsNewFile$splitSpecs$method # select from c(FALSE, TRUE)

glbObsDropCondition <- #NULL # : default
#   enclose in single-quotes b/c condition might include double qoutes
#       use | & ; NOT || &&    
#   '<condition>' 
    # 'grepl("^First Draft Video:", glbObsAll$Headline)'
    '(is.na(glbObsAll[, glb_rsp_var_raw]) & grepl("Train", glbObsAll[, glbFeatsId]))'
#nrow(do.call("subset",list(glbObsAll, parse(text=paste0("!(", glbObsDropCondition, ")")))))
    
glb_obs_repartition_train_condition <- NULL # : default
#    "<condition>" 

glb_max_fitobs <- NULL # or any integer
glbObsTrnPartitionSeed <- 123 # or any integer
                         
glb_is_regression <- TRUE; glb_is_classification <- !glb_is_regression; 
    glb_is_binomial <- NULL # or TRUE or FALSE

glb_rsp_var_raw <- "left_eye_center_x"

# for classification, the response variable has to be a factor
glb_rsp_var <- glb_rsp_var_raw # or "left_eye_center_x.fctr"

# if the response factor is based on numbers/logicals e.g (0/1 OR TRUE/FALSE vs. "A"/"B"), 
#   or contains spaces (e.g. "Not in Labor Force")
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- NULL 
# function(raw) {
#     return(raw ^ 0.5)
#     return(log(raw))
#     return(log(1 + raw))
#     return(log10(raw)) 
#     return(exp(-raw / 2))
#     ret_vals <- rep_len(NA, length(raw)); ret_vals[!is.na(raw)] <- ifelse(raw[!is.na(raw)] == 1, "Y", "N"); return(relevel(as.factor(ret_vals), ref="N"))
#     as.factor(paste0("B", raw))
#     as.factor(gsub(" ", "\\.", raw))    
#     }

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

# currently does not handle more than 1 column; consider concatenating multiple columns
# If glbFeatsId == NULL, ".rownames <- as.numeric(row.names())" is the default
glbFeatsId <- "ImageId" # choose from c(NULL : default, "<id_feat>") 
glbFeatsCategory <- NULL # choose from c(NULL : default, "<category_feat>")

# User-specified exclusions
glbFeatsExclude <- c(NULL
#   Feats that shd be excluded due to known causation by prediction variable
# , "<feat1", "<feat2>"
#   Feats that are linear combinations (alias in glm)
#   Feature-engineering phase -> start by excluding all features except id & category & work each one in
    # ,"left_eye_center_x"
    ,"left_eye_center_y"        
    ,"right_eye_center_x",        "right_eye_center_y"
    ,"left_eye_inner_corner_x",   "left_eye_inner_corner_y"  
    ,"left_eye_outer_corner_x",   "left_eye_outer_corner_y"  
    ,"right_eye_inner_corner_x",  "right_eye_inner_corner_y" 
    ,"right_eye_outer_corner_x",  "right_eye_outer_corner_y" 
    ,"left_eyebrow_inner_end_x",  "left_eyebrow_inner_end_y" 
    ,"left_eyebrow_outer_end_x",  "left_eyebrow_outer_end_y" 
    ,"right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y"
    ,"right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y"
    ,"nose_tip_x",                "nose_tip_y"               
    ,"mouth_left_corner_x",       "mouth_left_corner_y"      
    ,"mouth_right_corner_x",      "mouth_right_corner_y"     
    ,"mouth_center_top_lip_x",    "mouth_center_top_lip_y"   
    ,"mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"
    ,"Image"
                    ) 
if (glb_rsp_var_raw != glb_rsp_var)
    glbFeatsExclude <- union(glbFeatsExclude, glb_rsp_var_raw)                    

glbFeatsInteractionOnly <- list()
#glbFeatsInteractionOnly[["<child_feat>"]] <- "<parent_feat>"

glbFeatsDrop <- c(NULL
                # , "<feat1>", "<feat2>"
                ,"Image"
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
#myprint_df(data.frame(ImageId = mapfn(glbObsAll$.src, glbObsAll$.pos)))
#data.frame(ImageId = mapfn(glbObsAll$.src, glbObsAll$.pos))[7045:7055, ]

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
    # Some state acrnoyms need context for separation e.g. 
    #   LA/L.A. could either be "Louisiana" or "LosAngeles"
        # modRaw <- gsub("\\bL\\.A\\.( |,|')", "LosAngeles\\1", modRaw);
    #   OK/O.K. could either be "Oklahoma" or "Okay"
#         modRaw <- gsub("\\bACA OK\\b", "ACA OKay", modRaw); 
#         modRaw <- gsub("\\bNow O\\.K\\.\\b", "Now OKay", modRaw);        
    #   PR/P.R. could either be "PuertoRico" or "Public Relations"        
        # modRaw <- gsub("\\bP\\.R\\. Campaign", "PublicRelations Campaign", modRaw);        
    #   VA/V.A. could either be "Virginia" or "VeteransAdministration"        
        # modRaw <- gsub("\\bthe V\\.A\\.\\:", "the VeteranAffairs:", modRaw);
    #   
    # Custom mods

#         return(mod_raw) }

    # numeric
# Create feature based on record position/id in data   
glbFeatsDerive[[".pos"]] <- list(
    mapfn = function(.rnorm) { return(1:length(.rnorm)) }       
    , args = c(".rnorm"))    

glbFeatsDerive[["ImageId"]] <- list(
    mapfn = function(.src, .pos) { 
        # return(paste(.src, sprintf("%04d", .pos), sep = "#")) 
        return(paste(.src, sprintf("%04d", 
                                   ifelse(.src == "Train", .pos, .pos - 7049)
                                   ), sep = "#"))         
    }       
    , args = c(".src", ".pos")) 
#myprint_df(data.frame(ImageId = mapfn(glbObsAll$.src, glbObsAll$.pos)))
#data.frame(ImageId = mapfn(glbObsAll$.src, glbObsAll$.pos))[7045:7055, ]

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
#     mapfn = function(id, date) { return(paste(as.character(id), as.character(date), sep = "#")) }        
#     mapfn = function(PTS, oppPTS) { return(PTS - oppPTS) }
#     mapfn = function(startprice.log10.predict, startprice) {
#                  return(spdiff <- (10 ^ startprice.log10.predict) - startprice) } 
#     mapfn = function(productline, description) { as.factor(
#         paste(gsub(" ", "", productline), as.numeric(nchar(description) > 0), sep = "*")) }
#     mapfn = function(.src, .pos) { 
#         return(paste(.src, sprintf("%04d", 
#                                    ifelse(.src == "Train", .pos, .pos - 7049)
#                                    ), sep = "#")) }       

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

glbFeatsText <- list()
Sys.setlocale("LC_ALL", "C") # For english
```

```
## [1] "C/C/C/C/C/en_US.UTF-8"
```

```r
#glbFeatsText[["<TextFeature>"]] <- list(NULL,
#   ,names = myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL, 
#       <comma-separated-screened-names>
#   ))))
#   ,rareWords = myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL, 
#       <comma-separated-nonSCOWL-words>
#   ))))
#)

# Text Processing Step: custom modifications not present in txt_munge -> use glbFeatsDerive
# Text Processing Step: universal modifications
glb_txt_munge_filenames_pfx <- "<projectId>_mytxt_"

# Text Processing Step: tolower
# Text Processing Step: myreplacePunctuation
# Text Processing Step: removeWords
glb_txt_stop_words <- list()
# Remember to use unstemmed words
if (length(glbFeatsText) > 0) {
    require(tm)
    require(stringr)

    glb_txt_stop_words[["<txt_var>"]] <- sort(myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL
        # Remove any words from stopwords            
#         , setdiff(myreplacePunctuation(stopwords("english")), c("<keep_wrd1>", <keep_wrd2>"))
                                
        # Remove salutations
        ,"mr","mrs","dr","Rev"                                

        # Remove misc
        #,"th" # Happy [[:digit::]]+th birthday 

        # Remove terms present in Trn only or New only; search for "Partition post-stem"
        #   ,<comma-separated-terms>        

        # cor.y.train == NA
#         ,unlist(strsplit(paste(c(NULL
#           ,"<comma-separated-terms>"
#         ), collapse=",")

        # freq == 1; keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>

        # chisq.pval high (e.g. == 1); keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>

        # nzv.freqRatio high (e.g. >= glbFeatsNzvFreqMax); keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>        
                                            )))))
}
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^man", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])
#glbObsAll[glb_post_stem_words_terms_mtrx_lst[[txtFeat]][, 4866] > 0, c(glb_rsp_var, txtFeat)]

# To identify terms with a specific freq
#paste0(sort(subset(glb_post_stop_words_terms_df_lst[[txtFeat]], freq == 1)$term), collapse = ",")
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], freq <= 2)$term), collapse = ",")
#subset(glb_post_stem_words_terms_df_lst[[txtFeat]], term %in% c("zinger"))

# To identify terms with a specific freq & 
#   are not stemmed together later OR is value of color.fctr (e.g. gold)
#paste0(sort(subset(glb_post_stop_words_terms_df_lst[[txtFeat]], (freq == 1) & !(term %in% c("blacked","blemish","blocked","blocks","buying","cables","careful","carefully","changed","changing","chargers","cleanly","cleared","connect","connects","connected","contains","cosmetics","default","defaulting","defective","definitely","describe","described","devices","displays","drop","drops","engravement","excellant","excellently","feels","fix","flawlessly","frame","framing","gentle","gold","guarantee","guarantees","handled","handling","having","install","iphone","iphones","keeped","keeps","known","lights","line","lining","liquid","liquidation","looking","lots","manuals","manufacture","minis","most","mostly","network","networks","noted","opening","operated","performance","performs","person","personalized","photograph","physically","placed","places","powering","pre","previously","products","protection","purchasing","returned","rotate","rotation","running","sales","second","seconds","shipped","shuts","sides","skin","skinned","sticker","storing","thats","theres","touching","unusable","update","updates","upgrade","weeks","wrapped","verified","verify") ))$term), collapse = ",")

#print(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (freq <= 2)))
#glbObsAll[which(terms_mtrx[, 229] > 0), glbFeatsText]

# To identify terms with cor.y == NA
#orderBy(~-freq+term, subset(glb_post_stop_words_terms_df_lst[[txtFeat]], is.na(cor.y)))
#paste(sort(subset(glb_post_stop_words_terms_df_lst[[txtFeat]], is.na(cor.y))[, "term"]), collapse=",")
#orderBy(~-freq+term, subset(glb_post_stem_words_terms_df_lst[[txtFeat]], is.na(cor.y)))

# To identify terms with low cor.y.abs
#head(orderBy(~cor.y.abs+freq+term, subset(glb_post_stem_words_terms_df_lst[[txtFeat]], !is.na(cor.y))), 5)

# To identify terms with high chisq.pval
#subset(glb_post_stem_words_terms_df_lst[[txtFeat]], chisq.pval > 0.99)
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (chisq.pval > 0.99) & (freq <= 10))$term), collapse=",")
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (chisq.pval > 0.9))$term), collapse=",")
#head(orderBy(~-chisq.pval+freq+term, glb_post_stem_words_terms_df_lst[[txtFeat]]), 5)
#glbObsAll[glb_post_stem_words_terms_mtrx_lst[[txtFeat]][, 68] > 0, glbFeatsText]
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^m", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])

# To identify terms with high nzv.freqRatio
#summary(glb_post_stem_words_terms_df_lst[[txtFeat]]$nzv.freqRatio)
#paste0(sort(setdiff(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (nzv.freqRatio >= glbFeatsNzvFreqMax) & (freq < 10) & (chisq.pval >= 0.05))$term, c( "128gb","3g","4g","gold","ipad1","ipad3","ipad4","ipadair2","ipadmini2","manufactur","spacegray","sprint","tmobil","verizon","wifion"))), collapse=",")

# To identify obs with a txt term
#tail(orderBy(~-freq+term, glb_post_stop_words_terms_df_lst[[txtFeat]]), 20)
#mydspObs(list(descr.my.contains="non"), cols=c("color", "carrier", "cellular", "storage"))
#grep("ever", dimnames(terms_stop_mtrx)$Terms)
#which(terms_stop_mtrx[, grep("ipad", dimnames(terms_stop_mtrx)$Terms)] > 0)
#glbObsAll[which(terms_stop_mtrx[, grep("16", dimnames(terms_stop_mtrx)$Terms)[1]] > 0), c(glbFeatsCategory, "storage", txtFeat)]

# Text Processing Step: screen for names # Move to glbFeatsText specs section in order of text processing steps
# glbFeatsText[["<txtFeat>"]]$names <- myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL
#         # Person names for names screening
#         ,<comma-separated-list>
#         
#         # Company names
#         ,<comma-separated-list>
#                     
#         # Product names
#         ,<comma-separated-list>
#     ))))

# glbFeatsText[["<txtFeat>"]]$rareWords <- myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL
#         # Words not in SCOWL db
#         ,<comma-separated-list>
#     ))))

# To identify char vectors post glbFeatsTextMap
#grep("six(.*)hour", glb_txt_chr_lst[[txtFeat]], ignore.case = TRUE, value = TRUE)
#grep("[S|s]ix(.*)[H|h]our", glb_txt_chr_lst[[txtFeat]], value = TRUE)

# To identify whether terms shd be synonyms
#orderBy(~term, glb_post_stop_words_terms_df_lst[[txtFeat]][grep("^moder", glb_post_stop_words_terms_df_lst[[txtFeat]]$term), ])
# term_row_df <- glb_post_stop_words_terms_df_lst[[txtFeat]][grep("^came$", glb_post_stop_words_terms_df_lst[[txtFeat]]$term), ]
# 
# cor(glb_post_stop_words_terms_mtrx_lst[[txtFeat]][glbObsAll$.lcn == "Fit", term_row_df$pos], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")

# To identify which stopped words are "close" to a txt term
#sort(cluster_vars)

# Text Processing Step: stemDocument
# To identify stemmed txt terms
#glb_post_stop_words_terms_df_lst[[txtFeat]][grep("^la$", glb_post_stop_words_terms_df_lst[[txtFeat]]$term), ]
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^con", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])
#glbObsAll[which(terms_stem_mtrx[, grep("use", dimnames(terms_stem_mtrx)$Terms)[[1]]] > 0), c(glbFeatsId, "productline", txtFeat)]
#glbObsAll[which(TfIdf_stem_mtrx[, 191] > 0), c(glbFeatsId, glbFeatsCategory, txtFeat)]
#glbObsAll[which(glb_post_stop_words_terms_mtrx_lst[[txtFeat]][, 6165] > 0), c(glbFeatsId, glbFeatsCategory, txtFeat)]
#which(glbObsAll$UniqueID %in% c(11915, 11926, 12198))

# Text Processing Step: mycombineSynonyms
#   To identify which terms are associated with not -> combine "could not" & "couldn't"
#findAssocs(glb_full_DTM_lst[[txtFeat]], "not", 0.05)
#   To identify which synonyms should be combined
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^c", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])
chk_comb_cor <- function(syn_lst) {
#     cor(terms_stem_mtrx[glbObsAll$.src == "Train", grep("^(damag|dent|ding)$", dimnames(terms_stem_mtrx)[[2]])], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
    print(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], term %in% syn_lst$syns))
    print(subset(get_corpus_terms(tm_map(glbFeatsTextCorpus[[txtFeat]], mycombineSynonyms, list(syn_lst), lazy=FALSE)), term == syn_lst$word))
#     cor(terms_stop_mtrx[glbObsAll$.src == "Train", grep("^(damage|dent|ding)$", dimnames(terms_stop_mtrx)[[2]])], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
#     cor(rowSums(terms_stop_mtrx[glbObsAll$.src == "Train", grep("^(damage|dent|ding)$", dimnames(terms_stop_mtrx)[[2]])]), glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
}
#chk_comb_cor(syn_lst=list(word="cabl",  syns=c("cabl", "cord")))
#chk_comb_cor(syn_lst=list(word="damag",  syns=c("damag", "dent", "ding")))
#chk_comb_cor(syn_lst=list(word="dent",  syns=c("dent", "ding")))
#chk_comb_cor(syn_lst=list(word="use",  syns=c("use", "usag")))

glbFeatsTextSynonyms <- list()
# list parsed to collect glbFeatsText[[<txtFeat>]]$vldTerms
# glbFeatsTextSynonyms[["Hdln.my"]] <- list(NULL
#     # people in places
#     , list(word = "australia", syns = c("australia", "australian"))
#     , list(word = "italy", syns = c("italy", "Italian"))
#     , list(word = "newyork", syns = c("newyork", "newyorker"))    
#     , list(word = "Pakistan", syns = c("Pakistan", "Pakistani"))    
#     , list(word = "peru", syns = c("peru", "peruvian"))
#     , list(word = "qatar", syns = c("qatar", "qatari"))
#     , list(word = "scotland", syns = c("scotland", "scotish"))
#     , list(word = "Shanghai", syns = c("Shanghai", "Shanzhai"))    
#     , list(word = "venezuela", syns = c("venezuela", "venezuelan"))    
# 
#     # companies - needs to be data dependent 
#     #   - e.g. ensure BNP in this experiment/feat always refers to BNPParibas
#         
#     # general synonyms
#     , list(word = "Create", syns = c("Create","Creator")) 
#     , list(word = "cute", syns = c("cute","cutest"))     
#     , list(word = "Disappear", syns = c("Disappear","Fadeout"))     
#     , list(word = "teach", syns = c("teach", "taught"))     
#     , list(word = "theater",  syns = c("theater", "theatre", "theatres")) 
#     , list(word = "understand",  syns = c("understand", "understood"))    
#     , list(word = "weak",  syns = c("weak", "weaken", "weaker", "weakest"))
#     , list(word = "wealth",  syns = c("wealth", "wealthi"))    
#     
#     # custom synonyms (phrases)
#     
#     # custom synonyms (names)
#                                       )
#glbFeatsTextSynonyms[["<txtFeat>"]] <- list(NULL
#     , list(word="<stem1>",  syns=c("<stem1>", "<stem1_2>"))
#                                       )

for (txtFeat in names(glbFeatsTextSynonyms))
    for (entryIx in 1:length(glbFeatsTextSynonyms[[txtFeat]])) {
        glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$word <-
            str_to_lower(glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$word)
        glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$syns <-
            str_to_lower(glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$syns)        
    }        

glbFeatsTextSeed <- 181
# tm options include: check tm::weightSMART 
glb_txt_terms_control <- list( # Gather model performance & run-time stats
                    # weighting = function(x) weightSMART(x, spec = "nnn")
                    # weighting = function(x) weightSMART(x, spec = "lnn")
                    # weighting = function(x) weightSMART(x, spec = "ann")
                    # weighting = function(x) weightSMART(x, spec = "bnn")
                    # weighting = function(x) weightSMART(x, spec = "Lnn")
                    # 
                    weighting = function(x) weightSMART(x, spec = "ltn") # default
                    # weighting = function(x) weightSMART(x, spec = "lpn")                    
                    # 
                    # weighting = function(x) weightSMART(x, spec = "ltc")                    
                    # 
                    # weighting = weightBin 
                    # weighting = weightTf 
                    # weighting = weightTfIdf # : default
                # termFreq selection criteria across obs: tm default: list(global=c(1, Inf))
                    , bounds = list(global = c(1, Inf)) 
                # wordLengths selection criteria: tm default: c(3, Inf)
                    , wordLengths = c(1, Inf) 
                              ) 

glb_txt_cor_var <- glb_rsp_var # : default # or c(<feat>)

# select one from c("union.top.val.cor", "top.cor", "top.val", default: "top.chisq", "sparse")
glbFeatsTextFilter <- "top.chisq" 
glbFeatsTextTermsMax <- rep(10, length(glbFeatsText)) # :default
names(glbFeatsTextTermsMax) <- names(glbFeatsText)

# Text Processing Step: extractAssoc
glbFeatsTextAssocCor <- rep(1, length(glbFeatsText)) # :default 
names(glbFeatsTextAssocCor) <- names(glbFeatsText)

# Remember to use stemmed terms
glb_important_terms <- list()

# Text Processing Step: extractPatterns (ngrams)
glbFeatsTextPatterns <- list()
#glbFeatsTextPatterns[[<txtFeat>>]] <- list()
#glbFeatsTextPatterns[[<txtFeat>>]] <- c(metropolitan.diary.colon = "Metropolitan Diary:")

# Have to set it even if it is not used
# Properties:
#   numrows(glb_feats_df) << numrows(glbObsFit
#   Select terms that appear in at least 0.2 * O(FP/FN(glbObsOOB)) ???
#       numrows(glbObsOOB) = 1.1 * numrows(glbObsNew) ???
glb_sprs_thresholds <- NULL # or c(<txtFeat1> = 0.988, <txtFeat2> = 0.970, <txtFeat3> = 0.970)

glbFctrMaxUniqVals <- 20 # default: 20
glb_impute_na_data <- FALSE # or TRUE
glb_mice_complete.seed <- 144 # or any integer

glb_cluster <- FALSE # : default or TRUE
glb_cluster.seed <- 189 # or any integer
glb_cluster_entropy_var <- NULL # c(glb_rsp_var, as.factor(cut(glb_rsp_var, 3)), default: NULL)
glbFeatsTextClusterVarsExclude <- FALSE # default FALSE

glb_interaction_only_feats <- NULL # : default or c(<parent_feat> = "<child_feat>")

glbFeatsNzvFreqMax <- 19 # 19 : caret default
glbFeatsNzvUniqMin <- 10 # 10 : caret default

glbRFESizes <- list()
#glbRFESizes[["mdlFamily"]] <- c(4, 8, 16, 32, 64, 67, 68, 69) # Accuracy@69/70 = 0.8258

glbObsFitOutliers <- list()
# If outliers.n >= 10; consider concatenation of interaction vars
# glbObsFitOutliers[["<mdlFamily>"]] <- c(NULL
    # is.na(.rstudent)
    # is.na(.dffits)
    # .hatvalues >= 0.99        
    # -38,167,642 < minmax(.rstudent) < 49,649,823    
#     , <comma-separated-<glbFeatsId>>
#                                     )
glbObsTrnOutliers <- list()

# influence.measures: car::outlier; rstudent; dffits; hatvalues; dfbeta; dfbetas
#mdlId <- "RFE.X.glm"; obs_df <- fitobs_df
#mdlId <- "Final.glm"; obs_df <- trnobs_df
#mdlId <- "CSM2.X.glm"; obs_df <- fitobs_df
#print(outliers <- car::outlierTest(glb_models_lst[[mdlId]]$finalModel))
#mdlIdFamily <- paste0(head(unlist(str_split(mdlId, "\\.")), -1), collapse="."); obs_df <- dplyr::filter_(obs_df, interp(~(!(var %in% glbObsFitOutliers[[mdlIdFamily]])), var = as.name(glbFeatsId))); model_diags_df <- cbind(obs_df, data.frame(.rstudent=stats::rstudent(glb_models_lst[[mdlId]]$finalModel)), data.frame(.dffits=stats::dffits(glb_models_lst[[mdlId]]$finalModel)), data.frame(.hatvalues=stats::hatvalues(glb_models_lst[[mdlId]]$finalModel)));print(summary(model_diags_df[, c(".rstudent",".dffits",".hatvalues")])); table(cut(model_diags_df$.hatvalues, breaks=c(0.00, 0.98, 0.99, 1.00)))

#print(subset(model_diags_df, is.na(.rstudent))[, glbFeatsId])
#print(subset(model_diags_df, is.na(.dffits))[, glbFeatsId])
#print(model_diags_df[which.min(model_diags_df$.dffits), ])
#print(subset(model_diags_df, .hatvalues > 0.99)[, glbFeatsId])
#dffits_df <- merge(dffits_df, outliers_df, by="row.names", all.x=TRUE); row.names(dffits_df) <- dffits_df$Row.names; dffits_df <- subset(dffits_df, select=-Row.names)
#dffits_df <- merge(dffits_df, glbObsFit, by="row.names", all.x=TRUE); row.names(dffits_df) <- dffits_df$Row.names; dffits_df <- subset(dffits_df, select=-Row.names)
#subset(dffits_df, !is.na(.Bonf.p))

#mdlId <- "CSM.X.glm"; vars <- myextract_actual_feats(row.names(orderBy(reformulate(c("-", paste0(mdlId, ".imp"))), myget_feats_imp(glb_models_lst[[mdlId]])))); 
#model_diags_df <- glb_get_predictions(model_diags_df, mdlId, glb_rsp_var)
#obs_ix <- row.names(model_diags_df) %in% names(outliers$rstudent)[1]
#obs_ix <- which(is.na(model_diags_df$.rstudent))
#obs_ix <- which(is.na(model_diags_df$.dffits))
#myplot_parcoord(obs_df=model_diags_df[, c(glbFeatsId, glbFeatsCategory, ".rstudent", ".dffits", ".hatvalues", glb_rsp_var, paste0(glb_rsp_var, mdlId), vars[1:min(20, length(vars))])], obs_ix=obs_ix, id_var=glbFeatsId, category_var=glbFeatsCategory)

#model_diags_df[row.names(model_diags_df) %in% names(outliers$rstudent)[c(1:2)], ]
#ctgry_diags_df <- model_diags_df[model_diags_df[, glbFeatsCategory] %in% c("Unknown#0"), ]
#myplot_parcoord(obs_df=ctgry_diags_df[, c(glbFeatsId, glbFeatsCategory, ".rstudent", ".dffits", ".hatvalues", glb_rsp_var, "startprice.log10.predict.RFE.X.glmnet", indep_vars[1:20])], obs_ix=row.names(ctgry_diags_df) %in% names(outliers$rstudent)[1], id_var=glbFeatsId, category_var=glbFeatsCategory)
#table(glbObsFit[model_diags_df[, glbFeatsCategory] %in% c("iPad1#1"), "startprice.log10.cut.fctr"])
#glbObsFit[model_diags_df[, glbFeatsCategory] %in% c("iPad1#1"), c(glbFeatsId, "startprice")]

# No outliers & .dffits == NaN
#myplot_parcoord(obs_df=model_diags_df[, c(glbFeatsId, glbFeatsCategory, glb_rsp_var, "startprice.log10.predict.RFE.X.glmnet", indep_vars[1:10])], obs_ix=seq(1:nrow(model_diags_df))[is.na(model_diags_df$.dffits)], id_var=glbFeatsId, category_var=glbFeatsCategory)

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
glbMdlFamilies[["All.X"]] <- c("glmnet", "glm")  # non-NULL vector is mandatory
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
#glbMdlAllowParallel[["<mdlId>"]] <- FALSE
glbMdlAllowParallel[["All.X##rcv#glm"]] <- FALSE

# Check if tuning parameters make fit better; make it mdlFamily customizable ?
glbMdlTuneParams <- data.frame()
# When glmnet crashes at model$grid with error: ???
glmnetTuneParams <- rbind(data.frame()
                        ,data.frame(parameter = "alpha",  vals = "0.100 0.325 0.550 0.775 1.000")
                        ,data.frame(parameter = "lambda", vals = "9.342e-02")    
                        )
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams,
#                                cbind(data.frame(mdlId = "<mdlId>"),
#                                      glmnetTuneParams))

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

glb_sel_mdl_id <- "All.X##rcv#glmnet" #select from c(NULL, "All.X##rcv#glmnet", "RFE.X##rcv#glmnet", <mdlId>)
glb_fin_mdl_id <- NULL #select from c(NULL, glb_sel_mdl_id)

glb_dsp_cols <- c(glbFeatsId, glbFeatsCategory, glb_rsp_var
#               List critical cols excl. glbFeatsId, glbFeatsCategory & glb_rsp_var
                  )

# Output specs
glbOutDataVizFname <- NULL # choose from c(NULL, "<projectId>_obsall.csv")
glb_out_obs <- NULL # select from c(NULL : default to "new", "all", "new", "trn")
glb_out_vars_lst <- list()
# glbFeatsId will be the first output column, by default

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
glb_out_pfx <- "Faces_tmplt_"
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

![](Faces_tmplt_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_chunks_df <- myadd_chunk(NULL, "import.data")
```

```
##         label step_major step_minor label_minor    bgn end elapsed
## 1 import.data          1          0           0 10.714  NA      NA
```

## Step `1.0: import data`
#### chunk option: eval=<r condition>

```
## [1] "Reading file ./data/training/training.csv..."
## [1] "dimensions of data in ./data/training/training.csv: 7,049 rows x 31 cols"
## [1] "   Truncating Image to first 100 chars..."
##   left_eye_center_x left_eye_center_y right_eye_center_x
## 1          66.03356          39.00227           30.22701
## 2          64.33294          34.97008           29.94928
## 3          65.05705          34.90964           30.90379
## 4          65.22574          37.26177           32.02310
## 5          66.72530          39.62126           32.24481
## 6          69.68075          39.96875           29.18355
##   right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 1           36.42168                59.58208                39.64742
## 2           33.44871                58.85617                35.27435
## 3           34.90964                59.41200                36.32097
## 4           37.26177                60.00334                39.12718
## 5           38.04203                58.56589                39.62126
## 6           37.56336                62.86430                40.16927
##   left_eye_outer_corner_x left_eye_outer_corner_y right_eye_inner_corner_x
## 1                73.13035                39.97000                 36.35657
## 2                70.72272                36.18717                 36.03472
## 3                70.98442                36.32097                 37.67811
## 4                72.31471                38.38097                 37.61864
## 5                72.51593                39.88447                 36.98238
## 6                76.89824                41.17189                 36.40105
##   right_eye_inner_corner_y right_eye_outer_corner_x
## 1                 37.38940                 23.45287
## 2                 34.36153                 24.47251
## 3                 36.32097                 24.97642
## 4                 38.75411                 25.30727
## 5                 39.09485                 22.50611
## 6                 39.36763                 21.76553
##   right_eye_outer_corner_y left_eyebrow_inner_end_x
## 1                 37.38940                 56.95326
## 2                 33.14444                 53.98740
## 3                 36.60322                 55.74253
## 4                 38.00790                 56.43381
## 5                 38.30524                 57.24957
## 6                 38.56553                 59.76628
##   left_eyebrow_inner_end_y left_eyebrow_outer_end_x
## 1                 29.03365                 80.22713
## 2                 28.27595                 78.63421
## 3                 27.57095                 78.88737
## 4                 30.92986                 77.91026
## 5                 30.67218                 77.76294
## 6                 31.65129                 83.31364
##   left_eyebrow_outer_end_y right_eyebrow_inner_end_x
## 1                 32.22814                  40.22761
## 2                 30.40592                  42.72885
## 3                 32.65162                  42.19389
## 4                 31.66573                  41.67151
## 5                 31.73725                  38.03544
## 6                 35.35806                  39.40800
##   right_eyebrow_inner_end_y right_eyebrow_outer_end_x
## 1                  29.00232                  16.35638
## 2                  26.14604                  16.86536
## 3                  28.13545                  16.79116
## 4                  31.04999                  20.45802
## 5                  30.93538                  15.92587
## 6                  30.54639                  14.94908
##   right_eyebrow_outer_end_y nose_tip_x nose_tip_y mouth_left_corner_x
## 1                  29.64747   44.42057   57.06680            61.19531
## 2                  27.05886   48.20630   55.66094            56.42145
## 3                  32.08712   47.55726   53.53895            60.82295
## 4                  29.90934   51.88508   54.16654            65.59889
## 5                  30.67218   43.29953   64.88952            60.67141
## 6                  32.15013   52.46849   58.80000            64.86908
##   mouth_left_corner_y mouth_right_corner_x mouth_right_corner_y
## 1            79.97017             28.61450             77.38899
## 2            76.35200             35.12238             76.04766
## 3            73.01432             33.72632             72.73200
## 4            72.70372             37.24550             74.19548
## 5            77.52324             31.19175             76.99730
## 6            82.47118             31.99043             81.66908
##   mouth_center_top_lip_x mouth_center_top_lip_y mouth_center_bottom_lip_x
## 1               43.31260               72.93546                  43.13071
## 2               46.68460               70.26655                  45.46791
## 3               47.27495               70.19179                  47.27495
## 4               50.30317               70.09169                  51.56118
## 5               44.96275               73.70739                  44.22714
## 6               49.30811               78.48763                  49.43237
##   mouth_center_bottom_lip_y
## 1                  84.48577
## 2                  85.48017
## 3                  78.65937
## 4                  78.26838
## 5                  86.87117
## 6                  93.89877
##                                                                                                  Image
## 1 238 236 237 238 240 240 239 241 241 243 240 239 231 212 190 173 148 122 104 92 79 73 74 73 73 74 81 
## 2 219 215 204 196 204 211 212 200 180 168 178 196 194 196 203 209 199 192 197 201 207 215 199 190 182 
## 3 144 142 159 180 188 188 184 180 167 132 84 59 54 57 62 61 55 54 56 50 60 78 85 86 88 89 90 90 88 89 
## 4 193 192 193 194 194 194 193 192 168 111 50 12 1 1 1 1 1 1 1 1 1 1 6 16 19 17 13 13 16 22 25 31 34 27
## 5 147 148 160 196 215 214 216 217 219 220 206 188 166 104 88 81 77 71 63 58 58 52 58 62 59 60 55 51 57
## 6 167 169 170 167 156 145 106 68 52 24 20 15 21 14 6 9 11 11 29 49 61 71 76 80 82 84 84 84 83 88 91 92
##      left_eye_center_x left_eye_center_y right_eye_center_x
## 244           63.76497          38.17976           24.46709
## 1074          67.61736          35.73515           32.32068
## 3590          68.68139          35.63807           30.43863
## 5129          65.56684          37.63803           33.38537
## 5183          68.42459          43.83568           35.01755
## 6975          47.85052          37.39213           26.10544
##      right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 244            40.70400                57.27570                38.90097
## 1074           36.64797                61.53191                36.64797
## 3590           38.30054                      NA                      NA
## 5129           38.01223                      NA                      NA
## 5183           39.69364                      NA                      NA
## 6975           40.37090                      NA                      NA
##      left_eye_outer_corner_x left_eye_outer_corner_y
## 244                 71.69667                38.90097
## 1074                73.70281                35.43088
## 3590                      NA                      NA
## 5129                      NA                      NA
## 5183                      NA                      NA
## 6975                      NA                      NA
##      right_eye_inner_corner_x right_eye_inner_corner_y
## 244                  31.31697                 39.98279
## 1074                 38.71047                 37.25651
## 3590                       NA                       NA
## 5129                       NA                       NA
## 5183                       NA                       NA
## 6975                       NA                       NA
##      right_eye_outer_corner_x right_eye_outer_corner_y
## 244                  17.25661                 41.78501
## 1074                 25.62655                 36.95224
## 3590                       NA                       NA
## 5129                       NA                       NA
## 5183                       NA                       NA
## 6975                       NA                       NA
##      left_eyebrow_inner_end_x left_eyebrow_inner_end_y
## 244                  54.39086                 29.52766
## 1074                 59.73991                 31.37784
## 3590                       NA                       NA
## 5129                       NA                       NA
## 5183                       NA                       NA
## 6975                       NA                       NA
##      left_eyebrow_outer_end_x left_eyebrow_outer_end_y
## 244                  76.74434                 27.36403
## 1074                 77.04987                 28.43248
## 3590                       NA                       NA
## 5129                       NA                       NA
## 5183                       NA                       NA
## 6975                       NA                       NA
##      right_eyebrow_inner_end_x right_eyebrow_inner_end_y
## 244                   33.84121                  31.69049
## 1074                  43.01889                  32.12017
## 3590                        NA                        NA
## 5129                        NA                        NA
## 5183                        NA                        NA
## 6975                        NA                        NA
##      right_eyebrow_outer_end_x right_eyebrow_outer_end_y nose_tip_x
## 244                   11.84834                  33.49351   43.93573
## 1074                  21.97583                  32.08381   51.18638
## 3590                        NA                        NA   51.01229
## 5129                        NA                        NA   49.47608
## 5183                        NA                        NA   51.17158
## 6975                        NA                        NA   33.25451
##      nose_tip_y mouth_left_corner_x mouth_left_corner_y
## 244    52.96215            60.52034            76.75644
## 1074   57.33855            66.40000            72.24851
## 3590   59.60032                  NA                  NA
## 5129   63.08383                  NA                  NA
## 5183   64.96020                  NA                  NA
## 6975   59.13721                  NA                  NA
##      mouth_right_corner_x mouth_right_corner_y mouth_center_top_lip_x
## 244              35.64343             78.55946               46.81976
## 1074             36.27643             72.55285               51.49072
## 3590                   NA                   NA                     NA
## 5129                   NA                   NA                     NA
## 5183                   NA                   NA                     NA
## 6975                   NA                   NA                     NA
##      mouth_center_top_lip_y mouth_center_bottom_lip_x
## 244                70.62776                  47.90158
## 1074               72.24851                  51.49072
## 3590                     NA                  52.46453
## 5129                     NA                  49.32317
## 5183                     NA                  49.51476
## 6975                     NA                  38.31844
##      mouth_center_bottom_lip_y
## 244                   83.96773
## 1074                  80.76800
## 3590                  67.58770
## 5129                  72.81309
## 5183                  75.72958
## 6975                  76.71200
##                                                                                                     Image
## 244  41 36 34 33 41 47 43 38 37 39 40 35 27 23 27 31 32 28 26 29 35 38 37 39 42 41 40 42 41 39 44 50 51 4
## 1074 202 201 202 202 201 201 201 201 184 96 36 30 30 35 41 54 95 158 191 194 194 195 195 193 191 188 189 
## 3590 219 219 217 220 228 225 223 223 224 224 226 226 227 223 218 220 225 224 220 207 206 208 203 211 220 
## 5129 194 196 197 198 197 194 192 188 189 196 108 53 69 51 48 35 34 19 33 45 31 17 25 15 12 19 23 27 29 29
## 5183 140 137 127 118 111 104 105 111 115 116 117 111 104 99 93 90 93 96 93 90 91 95 101 114 126 137 150 1
## 6975 31 28 27 31 38 46 62 70 82 90 93 92 88 85 83 75 71 65 57 50 41 33 28 24 23 25 28 32 35 38 38 36 32 2
##      left_eye_center_x left_eye_center_y right_eye_center_x
## 7044          66.86722          37.35686           30.75093
## 7045          67.40255          31.84255           29.74675
## 7046          66.13440          38.36550           30.47863
## 7047          66.69073          36.84522           31.66642
## 7048          70.96508          39.85367           30.54328
## 7049          66.93831          43.42451           31.09606
##      right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 7044           40.11574                      NA                      NA
## 7045           38.63294                      NA                      NA
## 7046           39.95020                      NA                      NA
## 7047           39.68504                      NA                      NA
## 7048           40.77234                      NA                      NA
## 7049           39.52860                      NA                      NA
##      left_eye_outer_corner_x left_eye_outer_corner_y
## 7044                      NA                      NA
## 7045                      NA                      NA
## 7046                      NA                      NA
## 7047                      NA                      NA
## 7048                      NA                      NA
## 7049                      NA                      NA
##      right_eye_inner_corner_x right_eye_inner_corner_y
## 7044                       NA                       NA
## 7045                       NA                       NA
## 7046                       NA                       NA
## 7047                       NA                       NA
## 7048                       NA                       NA
## 7049                       NA                       NA
##      right_eye_outer_corner_x right_eye_outer_corner_y
## 7044                       NA                       NA
## 7045                       NA                       NA
## 7046                       NA                       NA
## 7047                       NA                       NA
## 7048                       NA                       NA
## 7049                       NA                       NA
##      left_eyebrow_inner_end_x left_eyebrow_inner_end_y
## 7044                       NA                       NA
## 7045                       NA                       NA
## 7046                       NA                       NA
## 7047                       NA                       NA
## 7048                       NA                       NA
## 7049                       NA                       NA
##      left_eyebrow_outer_end_x left_eyebrow_outer_end_y
## 7044                       NA                       NA
## 7045                       NA                       NA
## 7046                       NA                       NA
## 7047                       NA                       NA
## 7048                       NA                       NA
## 7049                       NA                       NA
##      right_eyebrow_inner_end_x right_eyebrow_inner_end_y
## 7044                        NA                        NA
## 7045                        NA                        NA
## 7046                        NA                        NA
## 7047                        NA                        NA
## 7048                        NA                        NA
## 7049                        NA                        NA
##      right_eyebrow_outer_end_x right_eyebrow_outer_end_y nose_tip_x
## 7044                        NA                        NA   43.54211
## 7045                        NA                        NA   48.26596
## 7046                        NA                        NA   47.91035
## 7047                        NA                        NA   49.46257
## 7048                        NA                        NA   50.75420
## 7049                        NA                        NA   47.06925
##      nose_tip_y mouth_left_corner_x mouth_left_corner_y
## 7044   64.94569                  NA                  NA
## 7045   67.02909                  NA                  NA
## 7046   66.62601                  NA                  NA
## 7047   67.51516                  NA                  NA
## 7048   66.72499                  NA                  NA
## 7049   73.03334                  NA                  NA
##      mouth_right_corner_x mouth_right_corner_y mouth_center_top_lip_x
## 7044                   NA                   NA                     NA
## 7045                   NA                   NA                     NA
## 7046                   NA                   NA                     NA
## 7047                   NA                   NA                     NA
## 7048                   NA                   NA                     NA
## 7049                   NA                   NA                     NA
##      mouth_center_top_lip_y mouth_center_bottom_lip_x
## 7044                     NA                  47.55504
## 7045                     NA                  50.42664
## 7046                     NA                  50.28740
## 7047                     NA                  49.46257
## 7048                     NA                  50.06519
## 7049                     NA                  45.90048
##      mouth_center_bottom_lip_y
## 7044                  79.49255
## 7045                  79.68392
## 7046                  77.98302
## 7047                  78.11712
## 7048                  79.58645
## 7049                  82.77310
##                                                                                                     Image
## 7044 150 150 132 63 44 74 86 61 62 57 44 70 93 115 114 115 99 110 94 108 108 94 97 86 79 75 101 90 93 89 
## 7045 71 74 85 105 116 128 139 150 170 187 201 209 218 219 212 198 184 181 185 188 193 196 199 202 206 208
## 7046 60 60 62 57 55 51 49 48 50 53 56 56 106 89 77 98 100 107 106 90 90 94 88 94 103 118 123 126 123 144 
## 7047 74 74 74 78 79 79 79 81 77 78 80 73 72 81 77 120 184 191 193 172 194 203 203 202 198 199 207 214 214
## 7048 254 254 254 254 254 238 193 145 121 118 119 109 106 106 105 107 109 111 113 117 126 129 129 129 129 
## 7049 53 62 67 76 86 91 97 105 105 106 107 108 112 117 123 129 130 128 132 134 136 142 149 155 157 157 153
## 'data.frame':	7049 obs. of  20 variables:
##  $ left_eye_center_x        : num  66 64.3 65.1 65.2 66.7 ...
##  $ left_eye_center_y        : num  39 35 34.9 37.3 39.6 ...
##  $ right_eye_center_x       : num  30.2 29.9 30.9 32 32.2 ...
##  $ right_eye_center_y       : num  36.4 33.4 34.9 37.3 38 ...
##  $ left_eye_inner_corner_x  : num  59.6 58.9 59.4 60 58.6 ...
##  $ left_eye_inner_corner_y  : num  39.6 35.3 36.3 39.1 39.6 ...
##  $ left_eye_outer_corner_x  : num  73.1 70.7 71 72.3 72.5 ...
##  $ left_eye_outer_corner_y  : num  40 36.2 36.3 38.4 39.9 ...
##  $ right_eye_inner_corner_x : num  36.4 36 37.7 37.6 37 ...
##  $ right_eye_inner_corner_y : num  37.4 34.4 36.3 38.8 39.1 ...
##  $ right_eye_outer_corner_x : num  23.5 24.5 25 25.3 22.5 ...
##  $ right_eye_outer_corner_y : num  37.4 33.1 36.6 38 38.3 ...
##  $ left_eyebrow_inner_end_x : num  57 54 55.7 56.4 57.2 ...
##  $ left_eyebrow_inner_end_y : num  29 28.3 27.6 30.9 30.7 ...
##  $ left_eyebrow_outer_end_x : num  80.2 78.6 78.9 77.9 77.8 ...
##  $ left_eyebrow_outer_end_y : num  32.2 30.4 32.7 31.7 31.7 ...
##  $ right_eyebrow_inner_end_x: num  40.2 42.7 42.2 41.7 38 ...
##  $ right_eyebrow_inner_end_y: num  29 26.1 28.1 31 30.9 ...
##  $ right_eyebrow_outer_end_x: num  16.4 16.9 16.8 20.5 15.9 ...
##  $ right_eyebrow_outer_end_y: num  29.6 27.1 32.1 29.9 30.7 ...
## NULL
## 'data.frame':	7049 obs. of  21 variables:
##  $ right_eye_outer_corner_x : num  23.5 24.5 25 25.3 22.5 ...
##  $ right_eye_outer_corner_y : num  37.4 33.1 36.6 38 38.3 ...
##  $ left_eyebrow_inner_end_x : num  57 54 55.7 56.4 57.2 ...
##  $ left_eyebrow_inner_end_y : num  29 28.3 27.6 30.9 30.7 ...
##  $ left_eyebrow_outer_end_x : num  80.2 78.6 78.9 77.9 77.8 ...
##  $ left_eyebrow_outer_end_y : num  32.2 30.4 32.7 31.7 31.7 ...
##  $ right_eyebrow_inner_end_x: num  40.2 42.7 42.2 41.7 38 ...
##  $ right_eyebrow_inner_end_y: num  29 26.1 28.1 31 30.9 ...
##  $ right_eyebrow_outer_end_x: num  16.4 16.9 16.8 20.5 15.9 ...
##  $ right_eyebrow_outer_end_y: num  29.6 27.1 32.1 29.9 30.7 ...
##  $ nose_tip_x               : num  44.4 48.2 47.6 51.9 43.3 ...
##  $ nose_tip_y               : num  57.1 55.7 53.5 54.2 64.9 ...
##  $ mouth_left_corner_x      : num  61.2 56.4 60.8 65.6 60.7 ...
##  $ mouth_left_corner_y      : num  80 76.4 73 72.7 77.5 ...
##  $ mouth_right_corner_x     : num  28.6 35.1 33.7 37.2 31.2 ...
##  $ mouth_right_corner_y     : num  77.4 76 72.7 74.2 77 ...
##  $ mouth_center_top_lip_x   : num  43.3 46.7 47.3 50.3 45 ...
##  $ mouth_center_top_lip_y   : num  72.9 70.3 70.2 70.1 73.7 ...
##  $ mouth_center_bottom_lip_x: num  43.1 45.5 47.3 51.6 44.2 ...
##  $ mouth_center_bottom_lip_y: num  84.5 85.5 78.7 78.3 86.9 ...
##  $ Image                    : chr  "238 236 237 238 240 240 239 241 241 243 240 239 231 212 190 173 148 122 104 92 79 73 74 73 73 74 81 74 60 64 75 86 93 102 100 1"| __truncated__ "219 215 204 196 204 211 212 200 180 168 178 196 194 196 203 209 199 192 197 201 207 215 199 190 182 180 183 190 190 176 175 175"| __truncated__ "144 142 159 180 188 188 184 180 167 132 84 59 54 57 62 61 55 54 56 50 60 78 85 86 88 89 90 90 88 89 91 94 95 98 99 101 104 107 "| __truncated__ "193 192 193 194 194 194 193 192 168 111 50 12 1 1 1 1 1 1 1 1 1 1 6 16 19 17 13 13 16 22 25 31 34 27 15 19 16 19 17 13 9 6 3 1 "| __truncated__ ...
## NULL
```

```
## Warning in myprint_str_df(df): [list output truncated]
```

```
## [1] "Reading file ./data/test/test.csv..."
## [1] "dimensions of data in ./data/test/test.csv: 1,783 rows x 2 cols"
## [1] "   Truncating Image to first 100 chars..."
##   ImageId
## 1       1
## 2       2
## 3       3
## 4       4
## 5       5
## 6       6
##                                                                                                  Image
## 1 182 183 182 182 180 180 176 169 156 137 124 103 79 62 54 56 58 48 49 45 39 37 42 43 52 61 78 93 104 
## 2 76 87 81 72 65 59 64 76 69 42 31 38 49 58 58 47 37 33 32 33 35 50 55 54 50 51 61 78 92 100 101 79 55
## 3 177 176 174 170 169 169 168 166 166 166 161 140 69 5 1 2 1 18 61 96 110 122 129 129 127 125 125 119 
## 4 176 174 174 175 174 174 176 176 175 171 165 157 143 134 134 137 138 137 135 135 134 137 135 128 128 
## 5 50 47 44 101 144 149 120 58 48 42 35 35 37 39 38 36 34 31 31 32 32 34 34 34 35 33 32 30 31 33 33 31 
## 6 177 177 177 171 142 115 97 84 89 90 88 82 63 51 40 35 39 37 42 38 29 35 43 64 95 117 127 115 108 125
##     ImageId
## 3         3
## 319     319
## 691     691
## 698     698
## 717     717
## 824     824
##                                                                                                    Image
## 3   177 176 174 170 169 169 168 166 166 166 161 140 69 5 1 2 1 18 61 96 110 122 129 129 127 125 125 119 
## 319 33 34 38 39 37 32 29 26 24 24 24 23 26 46 65 68 73 77 90 99 100 107 111 113 117 121 128 138 148 154 
## 691 34 32 34 43 38 23 8 15 18 19 19 39 47 45 30 43 51 50 44 40 37 36 37 37 37 39 41 43 48 50 53 57 59 62
## 698 14 14 15 16 18 21 23 25 27 29 30 31 31 33 34 36 39 45 60 73 81 89 97 108 115 121 126 128 129 127 124
## 717 17 14 21 20 17 40 77 93 103 121 150 165 153 144 144 118 90 112 132 155 167 170 172 176 181 179 182 1
## 824 86 110 151 194 223 197 177 158 149 144 181 207 216 206 185 163 142 128 117 109 83 53 54 57 63 71 80 
##      ImageId
## 1778    1778
## 1779    1779
## 1780    1780
## 1781    1781
## 1782    1782
## 1783    1783
##                                                                                                     Image
## 1778 100 106 105 106 105 104 104 108 112 114 111 108 108 111 113 111 108 117 130 114 114 135 108 87 91 82
## 1779 101 101 101 100 100 97 97 98 102 149 214 206 171 159 159 162 170 178 171 171 171 171 170 164 163 175
## 1780 201 191 171 158 145 140 136 130 123 115 108 104 100 96 99 115 132 155 167 174 170 160 159 158 166 17
## 1781 28 28 29 30 31 32 33 34 39 44 46 46 49 54 61 73 84 97 110 119 128 133 137 138 139 140 144 146 147 14
## 1782 104 95 71 57 46 52 65 70 70 67 76 72 69 69 72 75 73 68 81 67 58 35 33 41 27 20 13 28 39 53 70 75 80 
## 1783 63 61 64 66 66 64 65 70 69 70 77 83 63 34 22 21 21 18 23 12 17 22 24 37 32 15 15 20 20 15 9 9 9 8 9 
## 'data.frame':	1783 obs. of  2 variables:
##  $ ImageId: int  1 2 3 4 5 6 7 8 9 10 ...
##  $ Image  : chr  "182 183 182 182 180 180 176 169 156 137 124 103 79 62 54 56 58 48 49 45 39 37 42 43 52 61 78 93 104 107 114 115 117 122 120 122"| __truncated__ "76 87 81 72 65 59 64 76 69 42 31 38 49 58 58 47 37 33 32 33 35 50 55 54 50 51 61 78 92 100 101 79 55 47 52 50 47 39 38 52 46 25"| __truncated__ "177 176 174 170 169 169 168 166 166 166 161 140 69 5 1 2 1 18 61 96 110 122 129 129 127 125 125 119 112 110 111 107 102 102 99 "| __truncated__ "176 174 174 175 174 174 176 176 175 171 165 157 143 134 134 137 138 137 135 135 134 137 135 128 128 129 122 110 107 112 115 123"| __truncated__ ...
##  - attr(*, "comment")= chr "glbObsNew"
## NULL
```

```
## Warning: dropping vars: Image
```

```
## [1] "Creating new feature: .pos..."
## [1] "Creating new feature: ImageId..."
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
##   left_eye_center_x.cut.fctr  .src   .n
## 1                (46.7,70.7] Train 6652
## 2                       <NA>  Test 1783
## 3                (70.7,94.8] Train  342
## 4                (22.7,46.7] Train   45
## 5                       <NA> Train   10
##   left_eye_center_x.cut.fctr  .src   .n
## 1                (46.7,70.7] Train 6652
## 2                       <NA>  Test 1783
## 3                (70.7,94.8] Train  342
## 4                (22.7,46.7] Train   45
## 5                       <NA> Train   10
```

![](Faces_tmplt_files/figure-html/import.data-1.png) 

```
##    .src   .n
## 1 Train 7049
## 2  Test 1783
```

```
## [1] "Running glbObsDropCondition filter: (is.na(glbObsAll[, glb_rsp_var_raw]) & grepl(\"Train\", glbObsAll[, glbFeatsId]))"
## [1] "Partition stats:"
##   left_eye_center_x.cut.fctr  .src   .n
## 1                (46.7,70.7] Train 6652
## 2                       <NA>  Test 1783
## 3                (70.7,94.8] Train  342
## 4                (22.7,46.7] Train   45
##   left_eye_center_x.cut.fctr  .src   .n
## 1                (46.7,70.7] Train 6652
## 2                       <NA>  Test 1783
## 3                (70.7,94.8] Train  342
## 4                (22.7,46.7] Train   45
```

![](Faces_tmplt_files/figure-html/import.data-2.png) 

```
##    .src   .n
## 1 Train 7039
## 2  Test 1783
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
## 1  import.data          1          0           0 10.714 82.897  72.183
## 2 inspect.data          2          0           0 82.897     NA      NA
```

## Step `2.0: inspect data`

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

```
## Warning: Removed 1783 rows containing non-finite values (stat_bin).
```

![](Faces_tmplt_files/figure-html/inspect.data-1.png) 

```
## [1] "numeric data missing in : "
##         left_eye_center_x         left_eye_center_y 
##                      1783                      1783 
##        right_eye_center_x        right_eye_center_y 
##                      1789                      1789 
##   left_eye_inner_corner_x   left_eye_inner_corner_y 
##                      6556                      6556 
##   left_eye_outer_corner_x   left_eye_outer_corner_y 
##                      6559                      6559 
##  right_eye_inner_corner_x  right_eye_inner_corner_y 
##                      6559                      6559 
##  right_eye_outer_corner_x  right_eye_outer_corner_y 
##                      6559                      6559 
##  left_eyebrow_inner_end_x  left_eyebrow_inner_end_y 
##                      6557                      6557 
##  left_eyebrow_outer_end_x  left_eyebrow_outer_end_y 
##                      6603                      6603 
## right_eyebrow_inner_end_x right_eyebrow_inner_end_y 
##                      6561                      6561 
## right_eyebrow_outer_end_x right_eyebrow_outer_end_y 
##                      6594                      6594 
##                nose_tip_x                nose_tip_y 
##                      1783                      1783 
##       mouth_left_corner_x       mouth_left_corner_y 
##                      6562                      6562 
##      mouth_right_corner_x      mouth_right_corner_y 
##                      6561                      6561 
##    mouth_center_top_lip_x    mouth_center_top_lip_y 
##                      6557                      6557 
## mouth_center_bottom_lip_x mouth_center_bottom_lip_y 
##                      1816                      1816 
## [1] "numeric data w/ 0s in : "
## named integer(0)
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## ImageId 
##       0
```

![](Faces_tmplt_files/figure-html/inspect.data-2.png) 

```
## Warning: Removed 1783 rows containing non-finite values (stat_smooth).
```

```
## Warning: Removed 1783 rows containing non-finite values (stat_smooth).
```

```
## Warning: Removed 1783 rows containing missing values (geom_point).
```

```
## Warning: Removed 1783 rows containing non-finite values (stat_smooth).
```

```
## Warning: Removed 1783 rows containing non-finite values (stat_smooth).
```

```
## Warning: Removed 1783 rows containing missing values (geom_point).
```

![](Faces_tmplt_files/figure-html/inspect.data-3.png) 

```
##          label step_major step_minor label_minor    bgn   end elapsed
## 2 inspect.data          2          0           0 82.897 88.37   5.474
## 3   scrub.data          2          1           1 88.371    NA      NA
```

### Step `2.1: scrub data`

```
## [1] "numeric data missing in : "
##         left_eye_center_x         left_eye_center_y 
##                      1783                      1783 
##        right_eye_center_x        right_eye_center_y 
##                      1789                      1789 
##   left_eye_inner_corner_x   left_eye_inner_corner_y 
##                      6556                      6556 
##   left_eye_outer_corner_x   left_eye_outer_corner_y 
##                      6559                      6559 
##  right_eye_inner_corner_x  right_eye_inner_corner_y 
##                      6559                      6559 
##  right_eye_outer_corner_x  right_eye_outer_corner_y 
##                      6559                      6559 
##  left_eyebrow_inner_end_x  left_eyebrow_inner_end_y 
##                      6557                      6557 
##  left_eyebrow_outer_end_x  left_eyebrow_outer_end_y 
##                      6603                      6603 
## right_eyebrow_inner_end_x right_eyebrow_inner_end_y 
##                      6561                      6561 
## right_eyebrow_outer_end_x right_eyebrow_outer_end_y 
##                      6594                      6594 
##                nose_tip_x                nose_tip_y 
##                      1783                      1783 
##       mouth_left_corner_x       mouth_left_corner_y 
##                      6562                      6562 
##      mouth_right_corner_x      mouth_right_corner_y 
##                      6561                      6561 
##    mouth_center_top_lip_x    mouth_center_top_lip_y 
##                      6557                      6557 
## mouth_center_bottom_lip_x mouth_center_bottom_lip_y 
##                      1816                      1816 
## [1] "numeric data w/ 0s in : "
## named integer(0)
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## ImageId 
##       0
```

```
##            label step_major step_minor label_minor    bgn    end elapsed
## 3     scrub.data          2          1           1 88.371 89.996   1.626
## 4 transform.data          2          2           2 89.997     NA      NA
```

### Step `2.2: transform data`

```
##              label step_major step_minor label_minor    bgn    end elapsed
## 4   transform.data          2          2           2 89.997 90.037    0.04
## 5 extract.features          3          0           0 90.037     NA      NA
```

## Step `3.0: extract features`

```
##                     label step_major step_minor label_minor    bgn    end
## 5        extract.features          3          0           0 90.037 90.056
## 6 extract.features.string          3          1           1 90.057     NA
##   elapsed
## 5    0.02
## 6      NA
```

### Step `3.1: extract features string`

```
##                         label step_major step_minor label_minor    bgn end
## 1 extract.features.string.bgn          1          0           0 90.086  NA
##   elapsed
## 1      NA
```

```
##                                       label step_major step_minor
## 1               extract.features.string.bgn          1          0
## 2 extract.features.stringfactorize.str.vars          2          0
##   label_minor    bgn    end elapsed
## 1           0 90.086 90.096    0.01
## 2           0 90.096     NA      NA
```

```
##      .src   ImageId 
##    ".src" "ImageId"
```

```
##                       label step_major step_minor label_minor    bgn
## 6   extract.features.string          3          1           1 90.057
## 7 extract.features.datetime          3          2           2 90.110
##      end elapsed
## 6 90.109   0.053
## 7     NA      NA
```

### Step `3.2: extract features datetime`

```
##                           label step_major step_minor label_minor    bgn
## 1 extract.features.datetime.bgn          1          0           0 90.136
##   end elapsed
## 1  NA      NA
```

```
##                       label step_major step_minor label_minor    bgn
## 7 extract.features.datetime          3          2           2 90.110
## 8    extract.features.price          3          3           3 90.146
##      end elapsed
## 7 90.146   0.036
## 8     NA      NA
```

### Step `3.3: extract features price`

```
##                        label step_major step_minor label_minor    bgn end
## 1 extract.features.price.bgn          1          0           0 90.177  NA
##   elapsed
## 1      NA
```

```
##                    label step_major step_minor label_minor    bgn    end
## 8 extract.features.price          3          3           3 90.146 90.187
## 9  extract.features.text          3          4           4 90.187     NA
##   elapsed
## 8   0.041
## 9      NA
```

### Step `3.4: extract features text`

```
##                       label step_major step_minor label_minor    bgn end
## 1 extract.features.text.bgn          1          0           0 90.231  NA
##   elapsed
## 1      NA
```

```
##                    label step_major step_minor label_minor    bgn    end
## 9  extract.features.text          3          4           4 90.187 90.241
## 10  extract.features.end          3          5           5 90.242     NA
##    elapsed
## 9    0.054
## 10      NA
```

### Step `3.5: extract features end`

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](Faces_tmplt_files/figure-html/extract.features.end-1.png) 

```
##                   label step_major step_minor label_minor    bgn    end
## 10 extract.features.end          3          5           5 90.242 91.147
## 11  manage.missing.data          4          0           0 91.147     NA
##    elapsed
## 10   0.905
## 11      NA
```

### Step `4.0: manage missing data`

```
## [1] "numeric data missing in : "
##         left_eye_center_x         left_eye_center_y 
##                      1783                      1783 
##        right_eye_center_x        right_eye_center_y 
##                      1789                      1789 
##   left_eye_inner_corner_x   left_eye_inner_corner_y 
##                      6556                      6556 
##   left_eye_outer_corner_x   left_eye_outer_corner_y 
##                      6559                      6559 
##  right_eye_inner_corner_x  right_eye_inner_corner_y 
##                      6559                      6559 
##  right_eye_outer_corner_x  right_eye_outer_corner_y 
##                      6559                      6559 
##  left_eyebrow_inner_end_x  left_eyebrow_inner_end_y 
##                      6557                      6557 
##  left_eyebrow_outer_end_x  left_eyebrow_outer_end_y 
##                      6603                      6603 
## right_eyebrow_inner_end_x right_eyebrow_inner_end_y 
##                      6561                      6561 
## right_eyebrow_outer_end_x right_eyebrow_outer_end_y 
##                      6594                      6594 
##                nose_tip_x                nose_tip_y 
##                      1783                      1783 
##       mouth_left_corner_x       mouth_left_corner_y 
##                      6562                      6562 
##      mouth_right_corner_x      mouth_right_corner_y 
##                      6561                      6561 
##    mouth_center_top_lip_x    mouth_center_top_lip_y 
##                      6557                      6557 
## mouth_center_bottom_lip_x mouth_center_bottom_lip_y 
##                      1816                      1816 
## [1] "numeric data w/ 0s in : "
## named integer(0)
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## ImageId 
##       0
```

```
## [1] "numeric data missing in : "
##         left_eye_center_x         left_eye_center_y 
##                      1783                      1783 
##        right_eye_center_x        right_eye_center_y 
##                      1789                      1789 
##   left_eye_inner_corner_x   left_eye_inner_corner_y 
##                      6556                      6556 
##   left_eye_outer_corner_x   left_eye_outer_corner_y 
##                      6559                      6559 
##  right_eye_inner_corner_x  right_eye_inner_corner_y 
##                      6559                      6559 
##  right_eye_outer_corner_x  right_eye_outer_corner_y 
##                      6559                      6559 
##  left_eyebrow_inner_end_x  left_eyebrow_inner_end_y 
##                      6557                      6557 
##  left_eyebrow_outer_end_x  left_eyebrow_outer_end_y 
##                      6603                      6603 
## right_eyebrow_inner_end_x right_eyebrow_inner_end_y 
##                      6561                      6561 
## right_eyebrow_outer_end_x right_eyebrow_outer_end_y 
##                      6594                      6594 
##                nose_tip_x                nose_tip_y 
##                      1783                      1783 
##       mouth_left_corner_x       mouth_left_corner_y 
##                      6562                      6562 
##      mouth_right_corner_x      mouth_right_corner_y 
##                      6561                      6561 
##    mouth_center_top_lip_x    mouth_center_top_lip_y 
##                      6557                      6557 
## mouth_center_bottom_lip_x mouth_center_bottom_lip_y 
##                      1816                      1816 
## [1] "numeric data w/ 0s in : "
## named integer(0)
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## ImageId 
##       0
```

```
##                  label step_major step_minor label_minor    bgn    end
## 11 manage.missing.data          4          0           0 91.147 91.526
## 12        cluster.data          5          0           0 91.527     NA
##    elapsed
## 11    0.38
## 12      NA
```

## Step `5.0: cluster data`

```
##                      label step_major step_minor label_minor    bgn    end
## 12            cluster.data          5          0           0 91.527 91.578
## 13 partition.data.training          6          0           0 91.579     NA
##    elapsed
## 12   0.052
## 13      NA
```

## Step `6.0: partition data training`

```
## [1] "partition.data.training chunk: setup: elapsed: 0.00 secs"
```

```
## Loading required package: reshape2
```

```
## [1] "partition.data.training chunk: strata_mtrx complete: elapsed: 0.18 secs"
## [1] "partition.data.training chunk: obs_freq_df complete: elapsed: 0.18 secs"
```

```
## Loading required package: sampling
## 
## Attaching package: 'sampling'
## 
## The following objects are masked from 'package:survival':
## 
##     cluster, strata
## 
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## [1] "partition.data.training chunk: Fit/OOB partition complete: elapsed: 0.43 secs"
```

```
##   .category .n.Fit .n.OOB .n.Tst .freqRatio.Fit .freqRatio.OOB
## 1    .dummy   5076   1963   1783              1              1
##   .freqRatio.Tst
## 1              1
```

```
## [1] "glbObsAll: "
```

```
## [1] 8822   36
```

```
## [1] "glbObsTrn: "
```

```
## [1] 7039   36
```

```
## [1] "glbObsFit: "
```

```
## [1] 5076   35
```

```
## [1] "glbObsOOB: "
```

```
## [1] 1963   35
```

```
## [1] "glbObsNew: "
```

```
## [1] 1783   35
```

```
## [1] "partition.data.training chunk: teardown: elapsed: 0.71 secs"
```

```
##                      label step_major step_minor label_minor    bgn    end
## 13 partition.data.training          6          0           0 91.579 92.342
## 14         select.features          7          0           0 92.342     NA
##    elapsed
## 13   0.763
## 14      NA
```

## Step `7.0: select features`

```
## Warning in cor(data.matrix(entity_df[, sel_feats]), y =
## as.numeric(entity_df[, : the standard deviation is zero
```

```
##                                  cor.y exclude.as.feat   cor.y.abs
## left_eye_outer_corner_x    0.879976441               1 0.879976441
## left_eye_inner_corner_x    0.856572062               1 0.856572062
## left_eyebrow_outer_end_x   0.796637877               1 0.796637877
## left_eyebrow_inner_end_x   0.631805012               1 0.631805012
## nose_tip_x                 0.457467547               1 0.457467547
## mouth_left_corner_x        0.422891342               1 0.422891342
## mouth_center_bottom_lip_x  0.374314178               1 0.374314178
## right_eye_inner_corner_x   0.308648848               1 0.308648848
## mouth_left_corner_y        0.284087076               1 0.284087076
## right_eye_center_x         0.274459196               1 0.274459196
## right_eyebrow_inner_end_x  0.268462848               1 0.268462848
## mouth_center_bottom_lip_y  0.265655182               1 0.265655182
## mouth_center_top_lip_x     0.260752995               1 0.260752995
## mouth_right_corner_y       0.202841552               1 0.202841552
## nose_tip_y                 0.192274232               1 0.192274232
## mouth_center_top_lip_y     0.151302193               1 0.151302193
## left_eye_outer_corner_y    0.062128134               1 0.062128134
## .pos                       0.053615179               0 0.053615179
## left_eye_center_y          0.029907546               1 0.029907546
## left_eye_inner_corner_y    0.022211192               1 0.022211192
## mouth_right_corner_x      -0.008381762               1 0.008381762
## .rnorm                    -0.023380852               0 0.023380852
## right_eye_outer_corner_x  -0.065015870               1 0.065015870
## left_eyebrow_outer_end_y  -0.071141289               1 0.071141289
## right_eye_outer_corner_y  -0.075857186               1 0.075857186
## right_eye_inner_corner_y  -0.104697821               1 0.104697821
## left_eyebrow_inner_end_y  -0.128192861               1 0.128192861
## right_eyebrow_outer_end_x -0.137145707               1 0.137145707
## right_eye_center_y        -0.154727956               1 0.154727956
## right_eyebrow_outer_end_y -0.190547966               1 0.190547966
## right_eyebrow_inner_end_y -0.201853036               1 0.201853036
## .category                           NA               1          NA
##                           cor.high.X freqRatio percentUnique zeroVar   nzv
## left_eye_outer_corner_x           NA       1.0   31.97897429   FALSE FALSE
## left_eye_inner_corner_x           NA       1.0   32.07842023   FALSE FALSE
## left_eyebrow_outer_end_x          NA       1.0   31.43912488   FALSE FALSE
## left_eyebrow_inner_end_x          NA       1.0   32.10683336   FALSE FALSE
## nose_tip_x                        NA       1.0   99.55959653   FALSE FALSE
## mouth_left_corner_x               NA       1.0   31.95056116   FALSE FALSE
## mouth_center_bottom_lip_x         NA       1.0   98.99133400   FALSE FALSE
## right_eye_inner_corner_x          NA       2.5   31.92214803   FALSE FALSE
## mouth_left_corner_y               NA       1.0   31.96476772   FALSE FALSE
## right_eye_center_x                NA       1.0   99.77269499   FALSE FALSE
## right_eyebrow_inner_end_x         NA       1.0   31.99318085   FALSE FALSE
## mouth_center_bottom_lip_y         NA       1.0   99.20443245   FALSE FALSE
## mouth_center_top_lip_x            NA       1.0   32.06421367   FALSE FALSE
## mouth_right_corner_y              NA       1.0   31.95056116   FALSE FALSE
## nose_tip_y                        NA       1.0   99.73007529   FALSE FALSE
## mouth_center_top_lip_y            NA       1.0   32.06421367   FALSE FALSE
## left_eye_outer_corner_y           NA       1.0   31.87952834   FALSE FALSE
## .pos                              NA       1.0  100.00000000   FALSE FALSE
## left_eye_center_y                 NA       1.0   99.68745560   FALSE FALSE
## left_eye_inner_corner_y           NA       1.0   31.87952834   FALSE FALSE
## mouth_right_corner_x              NA       1.0   31.97897429   FALSE FALSE
## .rnorm                            NA       1.0  100.00000000   FALSE FALSE
## right_eye_outer_corner_x          NA       1.0   32.00738741   FALSE FALSE
## left_eyebrow_outer_end_y          NA       1.5   31.36809206   FALSE FALSE
## right_eye_outer_corner_y          NA       1.0   31.97897429   FALSE FALSE
## right_eye_inner_corner_y          NA       1.0   31.93635460   FALSE FALSE
## left_eyebrow_inner_end_y          NA       1.0   32.03580054   FALSE FALSE
## right_eyebrow_outer_end_x         NA       1.0   31.59539707   FALSE FALSE
## right_eye_center_y                NA       1.0   99.58800966   FALSE FALSE
## right_eyebrow_outer_end_y         NA       1.0   31.58119051   FALSE FALSE
## right_eyebrow_inner_end_y         NA       1.0   31.95056116   FALSE FALSE
## .category                         NA       0.0    0.01420656    TRUE  TRUE
##                           is.cor.y.abs.low
## left_eye_outer_corner_x              FALSE
## left_eye_inner_corner_x              FALSE
## left_eyebrow_outer_end_x             FALSE
## left_eyebrow_inner_end_x             FALSE
## nose_tip_x                           FALSE
## mouth_left_corner_x                  FALSE
## mouth_center_bottom_lip_x            FALSE
## right_eye_inner_corner_x             FALSE
## mouth_left_corner_y                  FALSE
## right_eye_center_x                   FALSE
## right_eyebrow_inner_end_x            FALSE
## mouth_center_bottom_lip_y            FALSE
## mouth_center_top_lip_x               FALSE
## mouth_right_corner_y                 FALSE
## nose_tip_y                           FALSE
## mouth_center_top_lip_y               FALSE
## left_eye_outer_corner_y              FALSE
## .pos                                 FALSE
## left_eye_center_y                    FALSE
## left_eye_inner_corner_y               TRUE
## mouth_right_corner_x                  TRUE
## .rnorm                               FALSE
## right_eye_outer_corner_x             FALSE
## left_eyebrow_outer_end_y             FALSE
## right_eye_outer_corner_y             FALSE
## right_eye_inner_corner_y             FALSE
## left_eyebrow_inner_end_y             FALSE
## right_eyebrow_outer_end_x            FALSE
## right_eye_center_y                   FALSE
## right_eyebrow_outer_end_y            FALSE
## right_eyebrow_inner_end_y            FALSE
## .category                               NA
```

```
## Warning in myplot_scatter(plt_feats_df, "percentUnique", "freqRatio",
## colorcol_name = "nzv", : converting nzv to class:factor
```

```
## Warning: Removed 31 rows containing missing values (geom_point).
```

```
## Warning: Removed 31 rows containing missing values (geom_point).
```

```
## Warning: Removed 31 rows containing missing values (geom_point).
```

![](Faces_tmplt_files/figure-html/select.features-1.png) 

```
##           cor.y exclude.as.feat cor.y.abs cor.high.X freqRatio
## .category    NA               1        NA         NA         0
##           percentUnique zeroVar  nzv is.cor.y.abs.low
## .category    0.01420656    TRUE TRUE               NA
```

![](Faces_tmplt_files/figure-html/select.features-2.png) 

```
## [1] "numeric data missing in : "
##         left_eye_center_x         left_eye_center_y 
##                      1783                      1783 
##        right_eye_center_x        right_eye_center_y 
##                      1789                      1789 
##   left_eye_inner_corner_x   left_eye_inner_corner_y 
##                      6556                      6556 
##   left_eye_outer_corner_x   left_eye_outer_corner_y 
##                      6559                      6559 
##  right_eye_inner_corner_x  right_eye_inner_corner_y 
##                      6559                      6559 
##  right_eye_outer_corner_x  right_eye_outer_corner_y 
##                      6559                      6559 
##  left_eyebrow_inner_end_x  left_eyebrow_inner_end_y 
##                      6557                      6557 
##  left_eyebrow_outer_end_x  left_eyebrow_outer_end_y 
##                      6603                      6603 
## right_eyebrow_inner_end_x right_eyebrow_inner_end_y 
##                      6561                      6561 
## right_eyebrow_outer_end_x right_eyebrow_outer_end_y 
##                      6594                      6594 
##                nose_tip_x                nose_tip_y 
##                      1783                      1783 
##       mouth_left_corner_x       mouth_left_corner_y 
##                      6562                      6562 
##      mouth_right_corner_x      mouth_right_corner_y 
##                      6561                      6561 
##    mouth_center_top_lip_x    mouth_center_top_lip_y 
##                      6557                      6557 
## mouth_center_bottom_lip_x mouth_center_bottom_lip_y 
##                      1816                      1816 
## [1] "numeric data w/ 0s in : "
## named integer(0)
## [1] "numeric data w/ Infs in : "
## named integer(0)
## [1] "numeric data w/ NaNs in : "
## named integer(0)
## [1] "string data missing in : "
## ImageId    .lcn 
##       0    1783
```

```
## [1] "glb_feats_df:"
```

```
## [1] 32 12
```

```
##                                  id exclude.as.feat rsp_var
## left_eye_center_x left_eye_center_x            TRUE    TRUE
```

```
##                                  id cor.y exclude.as.feat cor.y.abs
## left_eye_center_x left_eye_center_x    NA            TRUE        NA
##                   cor.high.X freqRatio percentUnique zeroVar nzv
## left_eye_center_x         NA        NA            NA      NA  NA
##                   is.cor.y.abs.low interaction.feat shapiro.test.p.value
## left_eye_center_x               NA               NA                   NA
##                   rsp_var_raw id_var rsp_var
## left_eye_center_x          NA     NA    TRUE
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
## 14 select.features          7          0           0 92.342 95.198   2.856
## 15      fit.models          8          0           0 95.198     NA      NA
```

## Step `8.0: fit models`

```r
fit.models_0_chunk_df <- myadd_chunk(NULL, "fit.models_0_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor   bgn end elapsed
## 1 fit.models_0_bgn          1          0       setup 95.74  NA      NA
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
if (glb_is_classification) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                                paste0("fit.models_0_", "MFO"), major.inc = FALSE,
                                        label.minor = "myMFO_classfr")

    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "MFO", type = glb_model_type, trainControl.method = "none",
        train.method = ifelse(glb_is_regression, "lm", "myMFO_classfr"))),
                            indep_vars = ".rnorm", rsp_var = glb_rsp_var,
                            fit_df = glbObsFit, OOB_df = glbObsOOB)

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

# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                        paste0("fit.models_0_", "Max.cor.Y.rcv.*X*"), major.inc = FALSE,
                                    label.minor = "glmnet")
```

```
##                            label step_major step_minor label_minor    bgn
## 1               fit.models_0_bgn          1          0       setup 95.740
## 2 fit.models_0_Max.cor.Y.rcv.*X*          1          1      glmnet 95.776
##      end elapsed
## 1 95.775   0.035
## 2     NA      NA
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
## [1] "    indep_vars: .pos,.rnorm"
```

```
## Loading required package: glmnet
## Loading required package: Matrix
## Loaded glmnet 2.0-2
```

```
## Fitting alpha = 0.1, lambda = 0.00376 on full training set
```

![](Faces_tmplt_files/figure-html/fit.models_0-1.png) 

```
##             Length Class      Mode     
## a0           61    -none-     numeric  
## beta        122    dgCMatrix  S4       
## df           61    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       61    -none-     numeric  
## dev.ratio    61    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
##  6.604155e+01  9.251347e-05 -8.055063e-02 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                           id       feats max.nTuningRuns
## 1 Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      0.759                 0.011  0.003546848
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     3.435571      0.003154002  0.003183645     3.458751      0.002166485
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
## [1] "    indep_vars: .pos,.rnorm"
```

```
## Loading required package: rpart
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.00498 on full training set
```

```
## Loading required package: rpart.plot
```

![](Faces_tmplt_files/figure-html/fit.models_0-2.png) ![](Faces_tmplt_files/figure-html/fit.models_0-3.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 5076 
## 
##            CP nsplit rel error
## 1 0.010138740      0 1.0000000
## 2 0.004984991      1 0.9898613
## 
## Variable importance
##   .pos .rnorm 
##     99      1 
## 
## Node number 1: 5076 observations,    complexity param=0.01013874
##   mean=66.36569, MSE=11.84516 
##   left son=2 (2400 obs) right son=3 (2676 obs)
##   Primary splits:
##       .pos   < 3326.5    to the left,  improve=0.010138740, (0 missing)
##       .rnorm < -1.918552 to the right, improve=0.001212581, (0 missing)
##   Surrogate splits:
##       .rnorm < -1.438368 to the left,  agree=0.534, adj=0.014, (0 split)
## 
## Node number 2: 2400 observations
##   mean=65.99976, MSE=11.91095 
## 
## Node number 3: 2676 observations
##   mean=66.69388, MSE=11.55836 
## 
## n= 5076 
## 
## node), split, n, deviance, yval
##       * denotes terminal node
## 
## 1) root 5076 60126.05 66.36569  
##   2) .pos< 3326.5 2400 28586.28 65.99976 *
##   3) .pos>=3326.5 2676 30930.17 66.69388 *
##                     id       feats max.nTuningRuns
## 1 Max.cor.Y##rcv#rpart .pos,.rnorm               5
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      1.724                 0.041   0.01013874
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1      3.43023               NA  0.009605254     3.447592               NA
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.008951045     0.09970171         0.00341234
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

if (length(glbFeatsText) > 0) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Txt.*"), major.inc = FALSE,
                                    label.minor = "glmnet")

    indepVars <- c(max_cor_y_x_vars)
    for (txtFeat in names(glbFeatsText))
        indepVars <- union(indepVars, 
            grep(paste(str_to_upper(substr(txtFeat, 1, 1)), "\\.(?!([T|P]\\.))", sep = ""),
                        names(glbObsAll), perl = TRUE, value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Text.nonTP", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        trainControl.allowParallel = glbMdlAllowParallel,                                
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)

    indepVars <- c(max_cor_y_x_vars)
    for (txtFeat in names(glbFeatsText))
        indepVars <- union(indepVars, 
            grep(paste(str_to_upper(substr(txtFeat, 1, 1)), "\\.T\\.", sep = ""),
                        names(glbObsAll), perl = TRUE, value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Text.onlyT", 
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

    indepVars <- c(max_cor_y_x_vars)
    for (txtFeat in names(glbFeatsText))
        indepVars <- union(indepVars, 
            grep(paste(str_to_upper(substr(txtFeat, 1, 1)), "\\.P\\.", sep = ""),
                        names(glbObsAll), perl = TRUE, value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Text.onlyP", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        trainControl.allowParallel = glbMdlAllowParallel,        
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

    ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
        id.prefix="Interact.High.cor.Y", 
        type=glb_model_type, trainControl.method="repeatedcv",
        trainControl.number=glb_rcv_n_folds, trainControl.repeats=glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
        train.method="glmnet")),
        indep_vars=c(max_cor_y_x_vars, paste(max_cor_y_x_vars[1], int_feats, sep=":")),
        rsp_var=glb_rsp_var, 
        fit_df=glbObsFit, OOB_df=glbObsOOB)
}    

# Low.cor.X
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                        paste0("fit.models_0_", "Low.cor.X"), major.inc = FALSE,
                                     label.minor = "glmnet")
```

```
##                            label step_major step_minor label_minor     bgn
## 2 fit.models_0_Max.cor.Y.rcv.*X*          1          1      glmnet  95.776
## 3         fit.models_0_Low.cor.X          1          2      glmnet 101.256
##       end elapsed
## 2 101.256    5.48
## 3      NA      NA
```

```r
indep_vars <- subset(glb_feats_df, is.na(cor.high.X) & !nzv & 
                              (exclude.as.feat != 1))[, "id"]  
indep_vars <- myadjust_interaction_feats(indep_vars)
ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
        id.prefix="Low.cor.X", 
        type=glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method="repeatedcv",
        trainControl.number=glb_rcv_n_folds, trainControl.repeats=glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
        train.method="glmnet")),
        indep_vars=indep_vars, rsp_var=glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
```

```
## [1] "fitting model: Low.cor.X##rcv#glmnet"
## [1] "    indep_vars: .pos,.rnorm"
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.1, lambda = 0.00376 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = "Low.cor.X", : model's bestTune found at an extreme of
## tuneGrid for parameter: alpha
```

![](Faces_tmplt_files/figure-html/fit.models_0-4.png) ![](Faces_tmplt_files/figure-html/fit.models_0-5.png) 

```
##             Length Class      Mode     
## a0           61    -none-     numeric  
## beta        122    dgCMatrix  S4       
## df           61    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       61    -none-     numeric  
## dev.ratio    61    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
##  6.604155e+01  9.251347e-05 -8.055063e-02 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                      id       feats max.nTuningRuns
## 1 Low.cor.X##rcv#glmnet .pos,.rnorm              20
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      2.065                 0.006  0.003546848
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     3.435697      0.003154002  0.003183645     3.458751      0.002166485
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.003675718      0.1022751        0.002502483
```

```r
fit.models_0_chunk_df <- 
    myadd_chunk(fit.models_0_chunk_df, "fit.models_0_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                    label step_major step_minor label_minor     bgn     end
## 3 fit.models_0_Low.cor.X          1          2      glmnet 101.256 104.931
## 4       fit.models_0_end          1          3    teardown 104.932      NA
##   elapsed
## 3   3.675
## 4      NA
```

```r
rm(ret_lst)

glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor     bgn    end elapsed
## 15 fit.models          8          0           0  95.198 104.95   9.752
## 16 fit.models          8          1           1 104.950     NA      NA
```


```r
fit.models_1_chunk_df <- myadd_chunk(NULL, "fit.models_1_bgn", label.minor="setup")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_1_bgn          1          0       setup 105.836  NA      NA
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
        fitobs_df <- glbObsFit[!(glbObsFit[, glbFeatsId] %in%
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
            trainControl.method = "repeatedcv", # or "none" if nominalWorkflow is crashing
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
## 1   fit.models_1_bgn          1          0       setup 105.836 105.847
## 2 fit.models_1_All.X          1          1       setup 105.847      NA
##   elapsed
## 1   0.011
## 2      NA
##                label step_major step_minor label_minor     bgn     end
## 2 fit.models_1_All.X          1          1       setup 105.847 105.854
## 3 fit.models_1_All.X          1          2      glmnet 105.855      NA
##   elapsed
## 2   0.007
## 3      NA
## [1] "fitting model: All.X##rcv#glmnet"
## [1] "    indep_vars: .pos,.rnorm"
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.1, lambda = 0.00376 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = mdl_id_pfx, : model's bestTune found at an extreme of
## tuneGrid for parameter: alpha
```

![](Faces_tmplt_files/figure-html/fit.models_1-1.png) ![](Faces_tmplt_files/figure-html/fit.models_1-2.png) 

```
##             Length Class      Mode     
## a0           61    -none-     numeric  
## beta        122    dgCMatrix  S4       
## df           61    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       61    -none-     numeric  
## dev.ratio    61    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
##  6.604155e+01  9.251347e-05 -8.055063e-02 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                  id       feats max.nTuningRuns min.elapsedtime.everything
## 1 All.X##rcv#glmnet .pos,.rnorm              20                       2.13
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit max.Adj.R.sq.fit
## 1                 0.007  0.003546848     3.435697      0.003154002
##   max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB max.Rsquared.fit
## 1  0.003183645     3.458751      0.002166485      0.003675718
##   min.RMSESD.fit max.RsquaredSD.fit
## 1      0.1022751        0.002502483
##                label step_major step_minor label_minor     bgn     end
## 3 fit.models_1_All.X          1          2      glmnet 105.855 109.517
## 4 fit.models_1_All.X          1          3         glm 109.517      NA
##   elapsed
## 3   3.662
## 4      NA
## [1] "fitting model: All.X##rcv#glm"
## [1] "    indep_vars: .pos,.rnorm"
## + Fold1.Rep1: parameter=none 
## - Fold1.Rep1: parameter=none 
## + Fold2.Rep1: parameter=none 
## - Fold2.Rep1: parameter=none 
## + Fold3.Rep1: parameter=none 
## - Fold3.Rep1: parameter=none 
## + Fold1.Rep2: parameter=none 
## - Fold1.Rep2: parameter=none 
## + Fold2.Rep2: parameter=none 
## - Fold2.Rep2: parameter=none 
## + Fold3.Rep2: parameter=none 
## - Fold3.Rep2: parameter=none 
## + Fold1.Rep3: parameter=none 
## - Fold1.Rep3: parameter=none 
## + Fold2.Rep3: parameter=none 
## - Fold2.Rep3: parameter=none 
## + Fold3.Rep3: parameter=none 
## - Fold3.Rep3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Faces_tmplt_files/figure-html/fit.models_1-3.png) ![](Faces_tmplt_files/figure-html/fit.models_1-4.png) ![](Faces_tmplt_files/figure-html/fit.models_1-5.png) ![](Faces_tmplt_files/figure-html/fit.models_1-6.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##      Min        1Q    Median        3Q       Max  
## -31.0235   -1.2536    0.1418    1.6532   28.2569  
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  6.604e+01  9.644e-02 684.762  < 2e-16 ***
## .pos         9.304e-05  2.374e-05   3.920 8.98e-05 ***
## .rnorm      -8.142e-02  4.809e-02  -1.693   0.0905 .  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for gaussian family taken to be 11.81013)
## 
##     Null deviance: 60126  on 5075  degrees of freedom
## Residual deviance: 59913  on 5073  degrees of freedom
## AIC: 26942
## 
## Number of Fisher Scoring iterations: 2
## 
##               id       feats max.nTuningRuns min.elapsedtime.everything
## 1 All.X##rcv#glm .pos,.rnorm               1                      1.154
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit min.aic.fit
## 1                  0.01  0.003547006     3.435704    26942.49
##   max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1       0.00315416  0.003181723     3.458754      0.002164562
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.003676082      0.1022627        0.002501139
```

```r
# Check if other preProcess methods improve model performance
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_preProc", major.inc = FALSE,
                label.minor = "preProc")
```

```
##                  label step_major step_minor label_minor     bgn     end
## 4   fit.models_1_All.X          1          3         glm 109.517 112.865
## 5 fit.models_1_preProc          1          4     preProc 112.865      NA
##   elapsed
## 4   3.348
## 5      NA
```

```r
mdl_id <- orderBy(get_model_sel_frmla(), glb_models_df)[1, "id"]
indep_vars_vctr <- trim(unlist(strsplit(glb_models_df[glb_models_df$id == mdl_id,
                                                      "feats"], "[,]")))
method <- tail(unlist(strsplit(mdl_id, "[.]")), 1)
mdl_id_pfx <- paste0(head(unlist(strsplit(mdl_id, "[.]")), -1), collapse = ".")
if (!is.null(glbObsFitOutliers[[mdl_id_pfx]])) {
    fitobs_df <- glbObsFit[!(glbObsFit[, glbFeatsId] %in%
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
##                                                    id       feats
## Max.cor.Y.rcv.1X1###glmnet Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm
## Max.cor.Y##rcv#rpart             Max.cor.Y##rcv#rpart .pos,.rnorm
## Low.cor.X##rcv#glmnet           Low.cor.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glmnet                   All.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glm                         All.X##rcv#glm .pos,.rnorm
##                            max.nTuningRuns min.elapsedtime.everything
## Max.cor.Y.rcv.1X1###glmnet               0                      0.759
## Max.cor.Y##rcv#rpart                     5                      1.724
## Low.cor.X##rcv#glmnet                   20                      2.065
## All.X##rcv#glmnet                       20                      2.130
## All.X##rcv#glm                           1                      1.154
##                            min.elapsedtime.final max.R.sq.fit min.RMSE.fit
## Max.cor.Y.rcv.1X1###glmnet                 0.011  0.003546848     3.435571
## Max.cor.Y##rcv#rpart                       0.041  0.010138740     3.430230
## Low.cor.X##rcv#glmnet                      0.006  0.003546848     3.435697
## All.X##rcv#glmnet                          0.007  0.003546848     3.435697
## All.X##rcv#glm                             0.010  0.003547006     3.435704
##                            max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB
## Max.cor.Y.rcv.1X1###glmnet      0.003154002  0.003183645     3.458751
## Max.cor.Y##rcv#rpart                     NA  0.009605254     3.447592
## Low.cor.X##rcv#glmnet           0.003154002  0.003183645     3.458751
## All.X##rcv#glmnet               0.003154002  0.003183645     3.458751
## All.X##rcv#glm                  0.003154160  0.003181723     3.458754
##                            max.Adj.R.sq.OOB max.Rsquared.fit
## Max.cor.Y.rcv.1X1###glmnet      0.002166485               NA
## Max.cor.Y##rcv#rpart                     NA      0.008951045
## Low.cor.X##rcv#glmnet           0.002166485      0.003675718
## All.X##rcv#glmnet               0.002166485      0.003675718
## All.X##rcv#glm                  0.002164562      0.003676082
##                            min.RMSESD.fit max.RsquaredSD.fit min.aic.fit
## Max.cor.Y.rcv.1X1###glmnet             NA                 NA          NA
## Max.cor.Y##rcv#rpart           0.09970171        0.003412340          NA
## Low.cor.X##rcv#glmnet          0.10227511        0.002502483          NA
## All.X##rcv#glmnet              0.10227511        0.002502483          NA
## All.X##rcv#glm                 0.10226265        0.002501139    26942.49
```

```r
rm(ret_lst)
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                  label step_major step_minor label_minor     bgn     end
## 5 fit.models_1_preProc          1          4     preProc 112.865 112.926
## 6     fit.models_1_end          1          5    teardown 112.927      NA
##   elapsed
## 5   0.062
## 6      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor     bgn     end elapsed
## 16 fit.models          8          1           1 104.950 112.936   7.986
## 17 fit.models          8          2           2 112.936      NA      NA
```


```r
fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_2_bgn          1          0       setup 114.862  NA      NA
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
##                                                    id       feats
## Max.cor.Y.rcv.1X1###glmnet Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm
## Max.cor.Y##rcv#rpart             Max.cor.Y##rcv#rpart .pos,.rnorm
## Low.cor.X##rcv#glmnet           Low.cor.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glmnet                   All.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glm                         All.X##rcv#glm .pos,.rnorm
##                            max.nTuningRuns max.R.sq.fit max.Adj.R.sq.fit
## Max.cor.Y.rcv.1X1###glmnet               0  0.003546848      0.003154002
## Max.cor.Y##rcv#rpart                     5  0.010138740               NA
## Low.cor.X##rcv#glmnet                   20  0.003546848      0.003154002
## All.X##rcv#glmnet                       20  0.003546848      0.003154002
## All.X##rcv#glm                           1  0.003547006      0.003154160
##                            max.R.sq.OOB max.Adj.R.sq.OOB max.Rsquared.fit
## Max.cor.Y.rcv.1X1###glmnet  0.003183645      0.002166485               NA
## Max.cor.Y##rcv#rpart        0.009605254               NA      0.008951045
## Low.cor.X##rcv#glmnet       0.003183645      0.002166485      0.003675718
## All.X##rcv#glmnet           0.003183645      0.002166485      0.003675718
## All.X##rcv#glm              0.003181723      0.002164562      0.003676082
##                            inv.elapsedtime.everything
## Max.cor.Y.rcv.1X1###glmnet                  1.3175231
## Max.cor.Y##rcv#rpart                        0.5800464
## Low.cor.X##rcv#glmnet                       0.4842615
## All.X##rcv#glmnet                           0.4694836
## All.X##rcv#glm                              0.8665511
##                            inv.elapsedtime.final inv.RMSE.fit inv.RMSE.OOB
## Max.cor.Y.rcv.1X1###glmnet              90.90909    0.2910724    0.2891217
## Max.cor.Y##rcv#rpart                    24.39024    0.2915257    0.2900575
## Low.cor.X##rcv#glmnet                  166.66667    0.2910617    0.2891217
## All.X##rcv#glmnet                      142.85714    0.2910617    0.2891217
## All.X##rcv#glm                         100.00000    0.2910612    0.2891215
##                             inv.aic.fit
## Max.cor.Y.rcv.1X1###glmnet           NA
## Max.cor.Y##rcv#rpart                 NA
## Low.cor.X##rcv#glmnet                NA
## All.X##rcv#glmnet                    NA
## All.X##rcv#glm             3.711609e-05
```

```r
# print(myplot_radar(radar_inp_df=plt_models_df))
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
## [1] "var:min.RMSESD.fit"
## [1] "var:max.RsquaredSD.fit"
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
## Warning: Removed 1 rows containing missing values (position_stack).
```

```
## Warning: Removed 4 rows containing missing values (geom_errorbar).
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
## Warning: Removed 1 rows containing missing values (position_stack).
```

```
## Warning: Removed 4 rows containing missing values (geom_errorbar).
```

![](Faces_tmplt_files/figure-html/fit.models_2-1.png) 

```r
dsp_models_cols <- c("id", 
                    glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                    grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
# if (glb_is_classification && glb_is_binomial) 
#     dsp_models_cols <- c(dsp_models_cols, "opt.prob.threshold.OOB")
print(dsp_models_df <- orderBy(get_model_sel_frmla(), glb_models_df)[, dsp_models_cols])
```

```
##                           id min.RMSE.OOB max.R.sq.OOB max.Adj.R.sq.fit
## 2       Max.cor.Y##rcv#rpart     3.447592  0.009605254               NA
## 1 Max.cor.Y.rcv.1X1###glmnet     3.458751  0.003183645      0.003154002
## 3      Low.cor.X##rcv#glmnet     3.458751  0.003183645      0.003154002
## 4          All.X##rcv#glmnet     3.458751  0.003183645      0.003154002
## 5             All.X##rcv#glm     3.458754  0.003181723      0.003154160
##   min.RMSE.fit
## 2     3.430230
## 1     3.435571
## 3     3.435697
## 4     3.435697
## 5     3.435704
```

```r
# print(myplot_radar(radar_inp_df = dsp_models_df))
print("Metrics used for model selection:"); print(get_model_sel_frmla())
```

```
## [1] "Metrics used for model selection:"
```

```
## ~+min.RMSE.OOB - max.R.sq.OOB - max.Adj.R.sq.fit + min.RMSE.fit
## <environment: 0x7ffb568076a8>
```

```r
print(sprintf("Best model id: %s", dsp_models_df[1, "id"]))
```

```
## [1] "Best model id: Max.cor.Y##rcv#rpart"
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
        #rowIx <- which.max(df[tp, predct_erabs_var_name]); df[tp, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a TN (true -ve), measure distance from 0.0
        tn <- which((df[, predct_var_name] == df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[1]))
        df[tn, predct_erabs_var_name] <- abs(0 - df[tn, predct_prob_var_name])
        #rowIx <- which.max(df[tn, predct_erabs_var_name]); df[tn, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a FP (flse +ve), measure distance from 0.0
        fp <- which((df[, predct_var_name] != df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[2]))
        df[fp, predct_erabs_var_name] <- abs(0 - df[fp, predct_prob_var_name])
        #rowIx <- which.max(df[fp, predct_erabs_var_name]); df[fp, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a FN (flse -ve), measure distance from 1.0
        fn <- which((df[, predct_var_name] != df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[1]))
        df[fn, predct_erabs_var_name] <- abs(1 - df[fn, predct_prob_var_name])
        #rowIx <- which.max(df[fn, predct_erabs_var_name]); df[fn, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]

        
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
## [1] "User specified selection: All.X##rcv#glmnet"
```

```r
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

![](Faces_tmplt_files/figure-html/fit.models_2-2.png) 

```
##             Length Class      Mode     
## a0           61    -none-     numeric  
## beta        122    dgCMatrix  S4       
## df           61    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       61    -none-     numeric  
## dev.ratio    61    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
##  6.604155e+01  9.251347e-05 -8.055063e-02 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"
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
## [1] "All.X##rcv#glmnet fit prediction diagnostics:"
```

```r
glbObsFit <- glb_get_predictions(df = glbObsFit, mdl_id = glb_sel_mdl_id, 
                                 rsp_var = glb_rsp_var)
print(sprintf("%s OOB prediction diagnostics:", glb_sel_mdl_id))
```

```
## [1] "All.X##rcv#glmnet OOB prediction diagnostics:"
```

```r
glbObsOOB <- glb_get_predictions(df = glbObsOOB, mdl_id = glb_sel_mdl_id, 
                                     rsp_var = glb_rsp_var)

print(glb_featsimp_df <- myget_feats_importance(mdl = glb_sel_mdl, featsimp_df = NULL))
```

```
##        All.X..rcv.glmnet.imp imp All.X##rcv#glmnet.imp
## .pos                     100 100                   100
## .rnorm                     0   0                     0
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
                        id_vars=glbFeatsId)
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
                                                id_vars = glbFeatsId,
                                                prob_threshold = prob_threshold))
    }    
}

if (glb_is_classification && glb_is_binomial)
    glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id, 
            prob_threshold = glb_models_df[glb_models_df$id == glb_sel_mdl_id, 
                                           "opt.prob.threshold.OOB"]) else
    glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id)                  
```

![](Faces_tmplt_files/figure-html/fit.models_2-3.png) ![](Faces_tmplt_files/figure-html/fit.models_2-4.png) 

```
##      left_eye_center_x left_eye_center_y right_eye_center_x
## 1908          22.76334          55.61720           1.528527
## 6406          37.95218          45.91296          30.872141
## 6316          38.85352          36.18525           3.672231
## 1621          40.45900          40.76164                 NA
## 2192          41.03546          37.92189          12.779046
##      right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 1908           56.40497                19.06495                56.29124
## 6406           45.09604                      NA                      NA
## 6316           30.39885                      NA                      NA
## 1621                 NA                35.55209                39.43738
## 2192           41.72817                32.98262                40.65430
##      left_eye_outer_corner_x left_eye_outer_corner_y
## 1908                27.57188                56.38438
## 6406                      NA                      NA
## 6316                      NA                      NA
## 1621                46.61831                42.57863
## 2192                48.63092                37.10972
##      right_eye_inner_corner_x right_eye_inner_corner_y
## 1908                 5.751046                 56.74390
## 6406                       NA                       NA
## 6316                       NA                       NA
## 1621                       NA                       NA
## 2192                18.194942                 42.77734
##      right_eye_outer_corner_x right_eye_outer_corner_y
## 1908                       NA                       NA
## 6406                       NA                       NA
## 6316                       NA                       NA
## 1621                       NA                       NA
## 2192                 4.424137                 42.36508
##      left_eyebrow_inner_end_x left_eyebrow_inner_end_y
## 1908                 17.88872                 51.04292
## 6406                       NA                       NA
## 6316                       NA                       NA
## 1621                 28.91893                 34.32552
## 2192                 24.27327                 33.97577
##      left_eyebrow_outer_end_x left_eyebrow_outer_end_y
## 1908                 32.20643                 49.89555
## 6406                       NA                       NA
## 6316                       NA                       NA
## 1621                 50.21866                 36.25489
## 2192                 54.04693                 30.76984
##      right_eyebrow_inner_end_x right_eyebrow_inner_end_y
## 1908                  6.921014                  51.24334
## 6406                        NA                        NA
## 6316                        NA                        NA
## 1621                 26.095147                  33.85276
## 2192                 12.013533                  37.27076
##      right_eyebrow_outer_end_x right_eyebrow_outer_end_y nose_tip_x
## 1908                        NA                        NA   12.94470
## 6406                        NA                        NA   34.13983
## 6316                        NA                        NA   15.24502
## 1621                   23.8719                  32.92002   19.53418
## 2192                        NA                        NA   20.51485
##      nose_tip_y mouth_left_corner_x mouth_left_corner_y
## 1908   60.06969            22.92336            73.48394
## 6406   51.26845                  NA                  NA
## 6316   63.26558                  NA                  NA
## 1621   55.67276                  NA                  NA
## 2192   64.67113            47.65531            75.03647
##      mouth_right_corner_x mouth_right_corner_y mouth_center_top_lip_x
## 1908             2.245766             74.12832               12.60517
## 6406                   NA                   NA                     NA
## 6316                   NA                   NA                     NA
## 1621                   NA                   NA               29.32441
## 2192            20.800524             79.31402               28.80664
##      mouth_center_top_lip_y mouth_center_bottom_lip_x
## 1908               71.88882                  12.53648
## 6406                     NA                  34.13983
## 6316                     NA                  18.25395
## 1621               66.69852                  29.32441
## 2192               81.62436                  28.85238
##      mouth_center_bottom_lip_y  .src    ImageId     .rnorm .pos .category
## 1908                  76.62886 Train Train#1908 -0.3524003 1908    .dummy
## 6406                  54.71770 Train Train#6406 -1.2134735 6406    .dummy
## 6316                  75.06980 Train Train#6316 -0.6224332 6316    .dummy
## 1621                  72.60010 Train Train#1621  0.9903102 1621    .dummy
## 2192                  83.08833 Train Train#2192 -0.3039093 2192    .dummy
##      left_eye_center_x.All.X..rcv.glmnet
## 1908                            66.24646
## 6406                            66.73194
## 6316                            66.67601
## 1621                            66.11175
## 2192                            66.26882
##      left_eye_center_x.All.X..rcv.glmnet.err
## 1908                                43.48311
## 6406                                28.77976
## 6316                                27.82248
## 1621                                25.65275
## 2192                                25.23336
##      left_eye_center_x.All.X..rcv.glmnet.err.abs
## 1908                                    43.48311
## 6406                                    28.77976
## 6316                                    27.82248
## 1621                                    25.65275
## 2192                                    25.23336
##      left_eye_center_x.All.X..rcv.glmnet.is.acc     .label
## 1908                                      FALSE Train#1908
## 6406                                      FALSE Train#6406
## 6316                                      FALSE Train#6316
## 1621                                      FALSE Train#1621
## 2192                                      FALSE Train#2192
```

![](Faces_tmplt_files/figure-html/fit.models_2-5.png) 

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
## .dummy    .dummy   1963   5076   1783              1              1
##        .freqRatio.Tst err.abs.fit.sum err.abs.fit.mean .n.fit
## .dummy              1        10653.64         2.098825   5076
##        err.abs.OOB.sum err.abs.OOB.mean
## .dummy        4143.806         2.110956
##           .n.OOB           .n.Fit           .n.Tst   .freqRatio.Fit 
##      1963.000000      5076.000000      1783.000000         1.000000 
##   .freqRatio.OOB   .freqRatio.Tst  err.abs.fit.sum err.abs.fit.mean 
##         1.000000         1.000000     10653.635212         2.098825 
##           .n.fit  err.abs.OOB.sum err.abs.OOB.mean 
##      5076.000000      4143.806047         2.110956
```

```r
write.csv(glbObsOOB[, c(glbFeatsId, 
                grep(glb_rsp_var, names(glbObsOOB), fixed=TRUE, value=TRUE))], 
    paste0(gsub(".", "_", paste0(glb_out_pfx, glb_sel_mdl_id), fixed=TRUE), 
           "_OOBobs.csv"), row.names=FALSE)

fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "teardown")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_2_bgn          1          0    teardown 120.722  NA      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc=FALSE)
```

```
##         label step_major step_minor label_minor     bgn     end elapsed
## 17 fit.models          8          2           2 112.936 120.731   7.795
## 18 fit.models          8          3           3 120.732      NA      NA
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

![](Faces_tmplt_files/figure-html/fit.models_3-1.png) 

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=TRUE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 18        fit.models          8          3           3 120.732 125.088
## 19 fit.data.training          9          0           0 125.089      NA
##    elapsed
## 18   4.356
## 19      NA
```

## Step `9.0: fit data training`

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
        glbObsTrn[!(glbObsTrn[, glbFeatsId] %in%
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
## [1] "fitting model: Final##rcv#glmnet"
## [1] "    indep_vars: .pos,.rnorm"
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.1, lambda = 0.0172 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = mdl_id_pfx, : model's bestTune found at an extreme of
## tuneGrid for parameter: alpha
```

![](Faces_tmplt_files/figure-html/fit.data.training_0-1.png) ![](Faces_tmplt_files/figure-html/fit.data.training_0-2.png) 

```
##             Length Class      Mode     
## a0           61    -none-     numeric  
## beta        122    dgCMatrix  S4       
## df           61    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       61    -none-     numeric  
## dev.ratio    61    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
##  6.604293e+01  8.993199e-05 -7.983577e-02 
## [1] "max lambda < lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
##  6.604253e+01  9.004627e-05 -8.002579e-02 
##                  id       feats max.nTuningRuns min.elapsedtime.everything
## 1 Final##rcv#glmnet .pos,.rnorm              20                      2.316
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit max.Adj.R.sq.fit
## 1                 0.008  0.003445728     3.441975      0.003162455
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.002829642      0.1253077        0.001332354
```

```r
rm(ret_lst)
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=FALSE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 19 fit.data.training          9          0           0 125.089 129.222
## 20 fit.data.training          9          1           1 129.223      NA
##    elapsed
## 19   4.133
## 20      NA
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
##        All.X..rcv.glmnet.imp All.X##rcv#glmnet.imp Final..rcv.glmnet.imp
## .pos                     100                   100                   100
## .rnorm                     0                     0                     0
##        imp Final##rcv#glmnet.imp
## .pos   100                   100
## .rnorm   0                     0
```

```r
if (glb_is_classification && glb_is_binomial)
    glb_analytics_diag_plots(obs_df=glbObsTrn, mdl_id=glb_fin_mdl_id, 
            prob_threshold=glb_models_df[glb_models_df$id == glb_sel_mdl_id, 
                                         "opt.prob.threshold.OOB"]) else
    glb_analytics_diag_plots(obs_df=glbObsTrn, mdl_id=glb_fin_mdl_id)                  
```

![](Faces_tmplt_files/figure-html/fit.data.training_1-1.png) ![](Faces_tmplt_files/figure-html/fit.data.training_1-2.png) 

```
##      left_eye_center_x left_eye_center_y right_eye_center_x
## 1908          22.76334          55.61720           1.528527
## 2788          35.34845          35.72961          10.797496
## 1862          36.16566          39.99390          23.196134
## 6493          36.94935          44.27019          34.291798
## 6406          37.95218          45.91296          30.872141
##      right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 1908           56.40497                19.06495                56.29124
## 2788           48.00511                      NA                      NA
## 1862           43.81474                34.14098                40.55803
## 6493           43.88362                      NA                      NA
## 6406           45.09604                      NA                      NA
##      left_eye_outer_corner_x left_eye_outer_corner_y
## 1908                27.57188                56.38438
## 2788                      NA                      NA
## 1862                40.60971                38.05205
## 6493                      NA                      NA
## 6406                      NA                      NA
##      right_eye_inner_corner_x right_eye_inner_corner_y
## 1908                 5.751046                 56.74390
## 2788                       NA                       NA
## 1862                25.041023                 43.77863
## 6493                       NA                       NA
## 6406                       NA                       NA
##      right_eye_outer_corner_x right_eye_outer_corner_y
## 1908                       NA                       NA
## 2788                       NA                       NA
## 1862                 22.04996                  43.9963
## 6493                       NA                       NA
## 6406                       NA                       NA
##      left_eyebrow_inner_end_x left_eyebrow_inner_end_y
## 1908                 17.88872                 51.04292
## 2788                       NA                       NA
## 1862                 24.54988                 37.64245
## 6493                       NA                       NA
## 6406                       NA                       NA
##      left_eyebrow_outer_end_x left_eyebrow_outer_end_y
## 1908                 32.20643                 49.89555
## 2788                       NA                       NA
## 1862                 43.18563                 32.53290
## 6493                       NA                       NA
## 6406                       NA                       NA
##      right_eyebrow_inner_end_x right_eyebrow_inner_end_y
## 1908                  6.921014                  51.24334
## 2788                        NA                        NA
## 1862                 22.136163                  42.76767
## 6493                        NA                        NA
## 6406                        NA                        NA
##      right_eyebrow_outer_end_x right_eyebrow_outer_end_y nose_tip_x
## 1908                        NA                        NA   12.94470
## 2788                        NA                        NA   28.28077
## 1862                   18.1551                   40.9182   21.35837
## 6493                        NA                        NA   35.40312
## 6406                        NA                        NA   34.13983
##      nose_tip_y mouth_left_corner_x mouth_left_corner_y
## 1908   60.06969            22.92336            73.48394
## 2788   60.28059                  NA                  NA
## 1862   66.08490            37.14547            78.11135
## 6493   46.20297                  NA                  NA
## 6406   51.26845                  NA                  NA
##      mouth_right_corner_x mouth_right_corner_y mouth_center_top_lip_x
## 1908             2.245766             74.12832               12.60517
## 2788                   NA                   NA                     NA
## 1862                   NA                   NA               29.67576
## 6493                   NA                   NA                     NA
## 6406                   NA                   NA                     NA
##      mouth_center_top_lip_y mouth_center_bottom_lip_x
## 1908               71.88882                  12.53648
## 2788                     NA                  34.97656
## 1862               78.16098                  33.16245
## 6493                     NA                  35.40312
## 6406                     NA                  34.13983
##      mouth_center_bottom_lip_y  .src    ImageId     .rnorm .pos .lcn
## 1908                  76.62886 Train Train#1908 -0.3524003 1908  OOB
## 2788                  72.55607 Train Train#2788 -0.8952096 2788  Fit
## 1862                  82.33012 Train Train#1862  0.6062188 1862  Fit
## 6493                  47.21769 Train Train#6493  0.7944260 6493  Fit
## 6406                  54.71770 Train Train#6406 -1.2134735 6406  OOB
##      .category left_eye_center_x.All.X..rcv.glmnet
## 1908    .dummy                                  NA
## 2788    .dummy                            66.37159
## 1862    .dummy                            66.16498
## 6493    .dummy                            66.57825
## 6406    .dummy                                  NA
##      left_eye_center_x.All.X..rcv.glmnet.err
## 1908                                      NA
## 2788                                31.02314
## 1862                                29.99932
## 6493                                29.62890
## 6406                                      NA
##      left_eye_center_x.All.X..rcv.glmnet.err.abs
## 1908                                          NA
## 2788                                    31.02314
## 1862                                    29.99932
## 6493                                    29.62890
## 6406                                          NA
##      left_eye_center_x.All.X..rcv.glmnet.is.acc
## 1908                                         NA
## 2788                                      FALSE
## 1862                                      FALSE
## 6493                                      FALSE
## 6406                                         NA
##      left_eye_center_x.Final..rcv.glmnet
## 1908                            66.24262
## 2788                            66.36516
## 1862                            66.16190
## 6493                            66.56350
## 6406                            66.71609
##      left_eye_center_x.Final..rcv.glmnet.err
## 1908                                43.47928
## 2788                                31.01671
## 1862                                29.99624
## 6493                                29.61414
## 6406                                28.76391
##      left_eye_center_x.Final..rcv.glmnet.err.abs
## 1908                                    43.47928
## 2788                                    31.01671
## 1862                                    29.99624
## 6493                                    29.61414
## 6406                                    28.76391
##      left_eye_center_x.Final..rcv.glmnet.is.acc     .label
## 1908                                      FALSE Train#1908
## 2788                                      FALSE Train#2788
## 1862                                      FALSE Train#1862
## 6493                                      FALSE Train#6493
## 6406                                      FALSE Train#6406
```

![](Faces_tmplt_files/figure-html/fit.data.training_1-3.png) 

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
## [1] "left_eye_center_x.Final..rcv.glmnet"        
## [2] "left_eye_center_x.Final..rcv.glmnet.err"    
## [3] "left_eye_center_x.Final..rcv.glmnet.err.abs"
## [4] "left_eye_center_x.Final..rcv.glmnet.is.acc"
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
        file = paste0(glb_out_pfx, "dsk.RData"))

#glb2Sav(); all.equal(savObsAll, glbObsAll); all.equal(sav_models_lst, glb_models_lst)
#load(file = paste0(glb_out_pfx, "dsk_knitr.RData"))
#cmpCols <- names(glbObsAll)[!grepl("\\.Final\\.", names(glbObsAll))]; all.equal(savObsAll[, cmpCols], glbObsAll[, cmpCols]); all.equal(savObsAll[, "H.P.http"], glbObsAll[, "H.P.http"]); 

replay.petrisim(pn = glb_analytics_pn, 
    replay.trans = (glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord = TRUE)
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

![](Faces_tmplt_files/figure-html/fit.data.training_1-4.png) 

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "predict.data.new", major.inc = TRUE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 20 fit.data.training          9          1           1 129.223 139.451
## 21  predict.data.new         10          0           0 139.452      NA
##    elapsed
## 20  10.229
## 21      NA
```

## Step `10.0: predict data new`

```
## Warning: Removed 1783 rows containing missing values (geom_point).
```

```
## Warning: Removed 1783 rows containing missing values (geom_point).
```

![](Faces_tmplt_files/figure-html/predict.data.new-1.png) 

```
## Warning: Removed 1783 rows containing missing values (geom_point).
```

```
## Warning: Removed 1783 rows containing missing values (geom_point).
```

![](Faces_tmplt_files/figure-html/predict.data.new-2.png) 

```
##      left_eye_center_x left_eye_center_y right_eye_center_x
## 7050                NA                NA                 NA
## 7051                NA                NA                 NA
## 7052                NA                NA                 NA
## 7053                NA                NA                 NA
## 7054                NA                NA                 NA
##      right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 7050                 NA                      NA                      NA
## 7051                 NA                      NA                      NA
## 7052                 NA                      NA                      NA
## 7053                 NA                      NA                      NA
## 7054                 NA                      NA                      NA
##      left_eye_outer_corner_x left_eye_outer_corner_y
## 7050                      NA                      NA
## 7051                      NA                      NA
## 7052                      NA                      NA
## 7053                      NA                      NA
## 7054                      NA                      NA
##      right_eye_inner_corner_x right_eye_inner_corner_y
## 7050                       NA                       NA
## 7051                       NA                       NA
## 7052                       NA                       NA
## 7053                       NA                       NA
## 7054                       NA                       NA
##      right_eye_outer_corner_x right_eye_outer_corner_y
## 7050                       NA                       NA
## 7051                       NA                       NA
## 7052                       NA                       NA
## 7053                       NA                       NA
## 7054                       NA                       NA
##      left_eyebrow_inner_end_x left_eyebrow_inner_end_y
## 7050                       NA                       NA
## 7051                       NA                       NA
## 7052                       NA                       NA
## 7053                       NA                       NA
## 7054                       NA                       NA
##      left_eyebrow_outer_end_x left_eyebrow_outer_end_y
## 7050                       NA                       NA
## 7051                       NA                       NA
## 7052                       NA                       NA
## 7053                       NA                       NA
## 7054                       NA                       NA
##      right_eyebrow_inner_end_x right_eyebrow_inner_end_y
## 7050                        NA                        NA
## 7051                        NA                        NA
## 7052                        NA                        NA
## 7053                        NA                        NA
## 7054                        NA                        NA
##      right_eyebrow_outer_end_x right_eyebrow_outer_end_y nose_tip_x
## 7050                        NA                        NA         NA
## 7051                        NA                        NA         NA
## 7052                        NA                        NA         NA
## 7053                        NA                        NA         NA
## 7054                        NA                        NA         NA
##      nose_tip_y mouth_left_corner_x mouth_left_corner_y
## 7050         NA                  NA                  NA
## 7051         NA                  NA                  NA
## 7052         NA                  NA                  NA
## 7053         NA                  NA                  NA
## 7054         NA                  NA                  NA
##      mouth_right_corner_x mouth_right_corner_y mouth_center_top_lip_x
## 7050                   NA                   NA                     NA
## 7051                   NA                   NA                     NA
## 7052                   NA                   NA                     NA
## 7053                   NA                   NA                     NA
## 7054                   NA                   NA                     NA
##      mouth_center_top_lip_y mouth_center_bottom_lip_x
## 7050                     NA                        NA
## 7051                     NA                        NA
## 7052                     NA                        NA
## 7053                     NA                        NA
## 7054                     NA                        NA
##      mouth_center_bottom_lip_y .src   ImageId     .rnorm .pos .lcn
## 7050                        NA Test Test#0001  0.1777388 7050     
## 7051                        NA Test Test#0002 -0.5182230 7051     
## 7052                        NA Test Test#0003 -0.5757552 7052     
## 7053                        NA Test Test#0004  0.6224964 7053     
## 7054                        NA Test Test#0005  1.0077015 7054     
##      .category left_eye_center_x.Final..rcv.glmnet
## 7050    .dummy                            66.66288
## 7051    .dummy                            66.71857
## 7052    .dummy                            66.72326
## 7053    .dummy                            66.62762
## 7054    .dummy                            66.59693
##      left_eye_center_x.Final..rcv.glmnet.err
## 7050                                      NA
## 7051                                      NA
## 7052                                      NA
## 7053                                      NA
## 7054                                      NA
##      left_eye_center_x.Final..rcv.glmnet.err.abs
## 7050                                          NA
## 7051                                          NA
## 7052                                          NA
## 7053                                          NA
## 7054                                          NA
##      left_eye_center_x.Final..rcv.glmnet.is.acc    .label
## 7050                                         NA Test#0001
## 7051                                         NA Test#0002
## 7052                                         NA Test#0003
## 7053                                         NA Test#0004
## 7054                                         NA Test#0005
```

![](Faces_tmplt_files/figure-html/predict.data.new-3.png) 

```
## Loading required package: stringr
```

```
## [1] "glb_sel_mdl_id: All.X##rcv#glmnet"
```

```
## [1] "glb_fin_mdl_id: Final##rcv#glmnet"
```

```
## [1] "Cross Validation issues:"
## Max.cor.Y.rcv.1X1###glmnet 
##                          0
```

```
##                            min.RMSE.OOB max.R.sq.OOB max.Adj.R.sq.fit
## Max.cor.Y##rcv#rpart           3.447592  0.009605254               NA
## Max.cor.Y.rcv.1X1###glmnet     3.458751  0.003183645      0.003154002
## Low.cor.X##rcv#glmnet          3.458751  0.003183645      0.003154002
## All.X##rcv#glmnet              3.458751  0.003183645      0.003154002
## All.X##rcv#glm                 3.458754  0.003181723      0.003154160
## Final##rcv#glmnet                    NA           NA      0.003162455
##                            min.RMSE.fit
## Max.cor.Y##rcv#rpart           3.430230
## Max.cor.Y.rcv.1X1###glmnet     3.435571
## Low.cor.X##rcv#glmnet          3.435697
## All.X##rcv#glmnet              3.435697
## All.X##rcv#glm                 3.435704
## Final##rcv#glmnet              3.441975
```

```
## [1] "All.X##rcv#glmnet OOB RMSE: 3.4588"
```

```
##        .freqRatio.Fit .freqRatio.OOB .freqRatio.Tst .n.Fit .n.OOB .n.Tst
## .dummy              1              1              1   5076   1963   1783
##        .n.fit .n.new .n.trn err.abs.OOB.mean err.abs.fit.mean
## .dummy   5076   1783   7039         2.110956         2.098825
##        err.abs.new.mean err.abs.trn.mean err.abs.OOB.sum err.abs.fit.sum
## .dummy               NA         2.102821        4143.806        10653.64
##        err.abs.new.sum err.abs.trn.sum
## .dummy              NA        14801.76
##   .freqRatio.Fit   .freqRatio.OOB   .freqRatio.Tst           .n.Fit 
##         1.000000         1.000000         1.000000      5076.000000 
##           .n.OOB           .n.Tst           .n.fit           .n.new 
##      1963.000000      1783.000000      5076.000000      1783.000000 
##           .n.trn err.abs.OOB.mean err.abs.fit.mean err.abs.new.mean 
##      7039.000000         2.110956         2.098825               NA 
## err.abs.trn.mean  err.abs.OOB.sum  err.abs.fit.sum  err.abs.new.sum 
##         2.102821      4143.806047     10653.635212               NA 
##  err.abs.trn.sum 
##     14801.757870 
##        .freqRatio.Fit .freqRatio.OOB .freqRatio.Tst .n.Fit .n.OOB .n.Tst
## .dummy              1              1              1   5076   1963   1783
##        .n.fit .n.new.x .n.new.y .n.trn err.abs.OOB.mean err.abs.fit.mean
## .dummy   5076     1783     1783   7039         2.110956         2.098825
##        err.abs.trn.mean err.abs.new.mean.x err.abs.new.mean.y
## .dummy         2.102821                 NA                 NA
##        err.abs.OOB.sum err.abs.fit.sum err.abs.trn.sum err.abs.new.sum.x
## .dummy        4143.806        10653.64        14801.76                NA
##        err.abs.new.sum.y
## .dummy                NA
##     .freqRatio.Fit     .freqRatio.OOB     .freqRatio.Tst 
##           1.000000           1.000000           1.000000 
##             .n.Fit             .n.OOB             .n.Tst 
##        5076.000000        1963.000000        1783.000000 
##             .n.fit           .n.new.x           .n.new.y 
##        5076.000000        1783.000000        1783.000000 
##             .n.trn   err.abs.OOB.mean   err.abs.fit.mean 
##        7039.000000           2.110956           2.098825 
##   err.abs.trn.mean err.abs.new.mean.x err.abs.new.mean.y 
##           2.102821                 NA                 NA 
##    err.abs.OOB.sum    err.abs.fit.sum    err.abs.trn.sum 
##        4143.806047       10653.635212       14801.757870 
##  err.abs.new.sum.x  err.abs.new.sum.y 
##                 NA                 NA
```

```
##      All.X..rcv.glmnet.imp All.X__rcv_glmnet.imp Final..rcv.glmnet.imp
## .pos                   100                   100                   100
##      Final__rcv_glmnet.imp
## .pos                   100
```

```
## [1] "glbObsNew prediction stats:"
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](Faces_tmplt_files/figure-html/predict.data.new-4.png) 

```
##                   label step_major step_minor label_minor     bgn     end
## 21     predict.data.new         10          0           0 139.452 154.481
## 22 display.session.info         11          0           0 154.482      NA
##    elapsed
## 21  15.029
## 22      NA
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
##                        label step_major step_minor label_minor     bgn
## 1                import.data          1          0           0  10.714
## 21          predict.data.new         10          0           0 139.452
## 20         fit.data.training          9          1           1 129.223
## 15                fit.models          8          0           0  95.198
## 16                fit.models          8          1           1 104.950
## 17                fit.models          8          2           2 112.936
## 2               inspect.data          2          0           0  82.897
## 18                fit.models          8          3           3 120.732
## 19         fit.data.training          9          0           0 125.089
## 14           select.features          7          0           0  92.342
## 3                 scrub.data          2          1           1  88.371
## 10      extract.features.end          3          5           5  90.242
## 13   partition.data.training          6          0           0  91.579
## 11       manage.missing.data          4          0           0  91.147
## 9      extract.features.text          3          4           4  90.187
## 6    extract.features.string          3          1           1  90.057
## 12              cluster.data          5          0           0  91.527
## 8     extract.features.price          3          3           3  90.146
## 4             transform.data          2          2           2  89.997
## 7  extract.features.datetime          3          2           2  90.110
## 5           extract.features          3          0           0  90.037
##        end elapsed duration
## 1   82.897  72.183   72.183
## 21 154.481  15.029   15.029
## 20 139.451  10.229   10.228
## 15 104.950   9.752    9.752
## 16 112.936   7.986    7.986
## 17 120.731   7.795    7.795
## 2   88.370   5.474    5.473
## 18 125.088   4.356    4.356
## 19 129.222   4.133    4.133
## 14  95.198   2.856    2.856
## 3   89.996   1.626    1.625
## 10  91.147   0.905    0.905
## 13  92.342   0.763    0.763
## 11  91.526   0.380    0.379
## 9   90.241   0.054    0.054
## 6   90.109   0.053    0.052
## 12  91.578   0.052    0.051
## 8   90.187   0.041    0.041
## 4   90.037   0.040    0.040
## 7   90.146   0.036    0.036
## 5   90.056   0.020    0.019
## [1] "Total Elapsed Time: 154.481 secs"
```

![](Faces_tmplt_files/figure-html/display.session.info-1.png) 
