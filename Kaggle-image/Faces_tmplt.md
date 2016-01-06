# Kaggle Facial Points Detection: Univ of Montreal: left_eye_center_x regression:: Faces_tmplt
bdanalytics  

**  **    
**Date: (Wed) Jan 06, 2016**    

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
glbFeatsCategory <- "Image.pxl.1.dgt.1" # choose from c(NULL : default, "<category_feat>")

# User-specified exclusions
glbFeatsExcludeLcl <- c(NULL
#   Required outputs
    ,"left_eye_center_x",         "left_eye_center_y"        
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
                    ) 
glbFeatsExclude <- c(NULL
#   Feats that shd be excluded due to known causation by prediction variable
# , "<feat1", "<feat2>"
#   Feats that are linear combinations (alias in glm)
#   Feature-engineering phase -> start by excluding all features except id & category & work each one in
    ,setdiff(glbFeatsExcludeLcl, glb_rsp_var_raw)
    ,"Image","Image.pxl.1.dgt.1"
                    ) 
if (glb_rsp_var_raw != glb_rsp_var)
    glbFeatsExclude <- union(glbFeatsExclude, glb_rsp_var_raw)                    

glbFeatsInteractionOnly <- list()
#glbFeatsInteractionOnly[["<child_feat>"]] <- "<parent_feat>"

glbFeatsDrop <- c(NULL
                # , "<feat1>", "<feat2>"
                # ,"Image"
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

glbFeatsDerive[["Image.pxl.1.dgt.1"]] <- list(
#     mapfn = function(Image) { return(cut(as.integer(sapply(Image, function(img) strsplit(img, " ")[[1]][1])),
#                                          breaks = 5)) }       
    mapfn = function(Image) { return(substr(Image, 1, 1)) }       
    , args = c("Image"))    

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
glbObsOut <- list(
                    # glbFeatsId will be the first output column, by default
                    vars = list()
                  )
glbOutDataVizFname <- NULL # choose from c(NULL, "<projectId>_obsall.csv")
glb_out_obs <- NULL # select from c(NULL : default to "new", "all", "new", "trn")

if (glb_is_classification && glb_is_binomial) {
    glbObsOut$vars[["Probability1"]] <- 
        "%<d-% glbObsNew[, mygetPredictIds(glb_rsp_var, glb_fin_mdl_id)$prob]" 
} else {
    glbObsOut$vars[[glbFeatsId]] <- 
        "%<d-% as.integer(gsub('Test#', '', glbObsNew[, glbFeatsId]))"
    glbObsOut$vars[[glb_rsp_var]] <- 
        "%<d-% glbObsNew[, mygetPredictIds(glb_rsp_var, glb_fin_mdl_id)$value]"
    for (outVar in setdiff(glbFeatsExcludeLcl, glb_rsp_var_raw))
        glbObsOut$vars[[outVar]] <- 
            paste0("%<d-% mean(glbObsAll[, \"", outVar, "\"], na.rm = TRUE)")
}    
# glbObsOut$vars[[glb_rsp_var_raw]] <- glb_rsp_var_raw
# glbObsOut$vars[[paste0(head(unlist(strsplit(mygetPredictIds$value, "")), -1), collapse = "")]] <-

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
##         label step_major step_minor label_minor   bgn end elapsed
## 1 import.data          1          0           0 8.084  NA      NA
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
## [1] "Creating new feature: .pos..."
## [1] "Creating new feature: ImageId..."
## [1] "Creating new feature: Image.pxl.1.dgt.1..."
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
## 1  import.data          1          0           0  8.084 72.245  64.161
## 2 inspect.data          2          0           0 72.246     NA      NA
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
##             Image           ImageId Image.pxl.1.dgt.1 
##                 0                 0                 0
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
##          label step_major step_minor label_minor    bgn    end elapsed
## 2 inspect.data          2          0           0 72.246 78.494   6.248
## 3   scrub.data          2          1           1 78.495     NA      NA
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
##             Image           ImageId Image.pxl.1.dgt.1 
##                 0                 0                 0
```

```
##            label step_major step_minor label_minor    bgn    end elapsed
## 3     scrub.data          2          1           1 78.495 80.117   1.622
## 4 transform.data          2          2           2 80.118     NA      NA
```

### Step `2.2: transform data`

```
##              label step_major step_minor label_minor    bgn   end elapsed
## 4   transform.data          2          2           2 80.118 80.16   0.042
## 5 extract.features          3          0           0 80.160    NA      NA
```

## Step `3.0: extract features`

```
##                     label step_major step_minor label_minor    bgn   end
## 5        extract.features          3          0           0 80.160 80.18
## 6 extract.features.string          3          1           1 80.181    NA
##   elapsed
## 5    0.02
## 6      NA
```

### Step `3.1: extract features string`

```
##                         label step_major step_minor label_minor   bgn end
## 1 extract.features.string.bgn          1          0           0 80.21  NA
##   elapsed
## 1      NA
```

```
##                                       label step_major step_minor
## 1               extract.features.string.bgn          1          0
## 2 extract.features.stringfactorize.str.vars          2          0
##   label_minor   bgn   end elapsed
## 1           0 80.21 80.22    0.01
## 2           0 80.22    NA      NA
```

```
##               Image                .src             ImageId 
##             "Image"              ".src"           "ImageId" 
##   Image.pxl.1.dgt.1 
## "Image.pxl.1.dgt.1"
```

```
##                       label step_major step_minor label_minor    bgn
## 6   extract.features.string          3          1           1 80.181
## 7 extract.features.datetime          3          2           2 80.235
##      end elapsed
## 6 80.234   0.053
## 7     NA      NA
```

### Step `3.2: extract features datetime`

```
##                           label step_major step_minor label_minor    bgn
## 1 extract.features.datetime.bgn          1          0           0 80.262
##   end elapsed
## 1  NA      NA
```

```
##                       label step_major step_minor label_minor    bgn
## 7 extract.features.datetime          3          2           2 80.235
## 8    extract.features.price          3          3           3 80.273
##      end elapsed
## 7 80.273   0.038
## 8     NA      NA
```

### Step `3.3: extract features price`

```
##                        label step_major step_minor label_minor    bgn end
## 1 extract.features.price.bgn          1          0           0 80.297  NA
##   elapsed
## 1      NA
```

```
##                    label step_major step_minor label_minor    bgn    end
## 8 extract.features.price          3          3           3 80.273 80.307
## 9  extract.features.text          3          4           4 80.307     NA
##   elapsed
## 8   0.034
## 9      NA
```

### Step `3.4: extract features text`

```
##                       label step_major step_minor label_minor    bgn end
## 1 extract.features.text.bgn          1          0           0 80.351  NA
##   elapsed
## 1      NA
```

```
##                    label step_major step_minor label_minor    bgn    end
## 9  extract.features.text          3          4           4 80.307 80.362
## 10  extract.features.end          3          5           5 80.362     NA
##    elapsed
## 9    0.055
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
## 10 extract.features.end          3          5           5 80.362 81.265
## 11  manage.missing.data          4          0           0 81.265     NA
##    elapsed
## 10   0.903
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
##             Image           ImageId Image.pxl.1.dgt.1 
##                 0                 0                 0
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
##             Image           ImageId Image.pxl.1.dgt.1 
##                 0                 0                 0
```

```
##                  label step_major step_minor label_minor    bgn    end
## 11 manage.missing.data          4          0           0 81.265 81.652
## 12        cluster.data          5          0           0 81.653     NA
##    elapsed
## 11   0.387
## 12      NA
```

## Step `5.0: cluster data`

```
##                      label step_major step_minor label_minor    bgn    end
## 12            cluster.data          5          0           0 81.653 81.715
## 13 partition.data.training          6          0           0 81.715     NA
##    elapsed
## 12   0.062
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
## [1] "Prediction Hints by Catgeory:"
##   Image.pxl.1.dgt.1 left_eye_center_x.cut.fctr.(22.8,65.1]
## 1                 0                                     13
##   left_eye_center_x.cut.fctr.(65.1,66.4]
## 1                                     11
##   left_eye_center_x.cut.fctr.(66.4,66.5]
## 1                                     NA
##   left_eye_center_x.cut.fctr.(66.5,68]
## 1                                    8
##   left_eye_center_x.cut.fctr.(68,94.7] .n.tst .strata.(22.8,65.1]
## 1                                   12     15                   5
##   .strata.(65.1,66.4] .strata.(66.4,66.5]
## 1                   4                   1
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
## [1] "partition.data.training chunk: Fit/OOB partition complete: elapsed: 0.95 secs"
```

```
##    Image.pxl.1.dgt.1 .n.Fit .n.OOB .n.Tst .freqRatio.Fit .freqRatio.OOB
## 2                  1   2000    744    674     0.39572616     0.37481108
## 3                  2   1078    434    393     0.21329640     0.21863980
## 4                  3    323    153    137     0.06390977     0.07707809
## 5                  4    329    132    117     0.06509695     0.06649874
## 6                  5    263    129    115     0.05203799     0.06498741
## 7                  6    263    109     97     0.05203799     0.05491184
## 8                  7    271     97     86     0.05362089     0.04886650
## 10                 9    233     89     79     0.04610210     0.04483627
## 9                  8    268     80     70     0.05302731     0.04030227
## 1                  0     26     18     15     0.00514444     0.00906801
##    .freqRatio.Tst
## 2     0.378014582
## 3     0.220415031
## 4     0.076836792
## 5     0.065619742
## 6     0.064498037
## 7     0.054402692
## 8     0.048233315
## 10    0.044307347
## 9     0.039259675
## 1     0.008412787
```

```
## [1] "glbObsAll: "
```

```
## [1] 8822   37
```

```
## [1] "glbObsTrn: "
```

```
## [1] 7039   37
```

```
## [1] "glbObsFit: "
```

```
## [1] 5054   36
```

```
## [1] "glbObsOOB: "
```

```
## [1] 1985   36
```

```
## [1] "glbObsNew: "
```

```
## [1] 1783   36
```

```
## [1] "partition.data.training chunk: teardown: elapsed: 2.85 secs"
```

```
##                      label step_major step_minor label_minor    bgn    end
## 13 partition.data.training          6          0           0 81.715 84.628
## 14         select.features          7          0           0 84.629     NA
##    elapsed
## 13   2.914
## 14      NA
```

## Step `7.0: select features`

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
##                           cor.high.X freqRatio percentUnique zeroVar   nzv
## left_eye_outer_corner_x           NA       1.0      31.99318   FALSE FALSE
## left_eye_inner_corner_x           NA       1.0      32.14945   FALSE FALSE
## left_eyebrow_outer_end_x          NA       1.0      31.45333   FALSE FALSE
## left_eyebrow_inner_end_x          NA       1.0      32.12104   FALSE FALSE
## nose_tip_x                        NA       1.0      99.82952   FALSE FALSE
## mouth_left_corner_x               NA       1.0      31.95056   FALSE FALSE
## mouth_center_bottom_lip_x         NA       1.0      99.26126   FALSE FALSE
## right_eye_inner_corner_x          NA       2.5      31.92215   FALSE FALSE
## mouth_left_corner_y               NA       1.0      31.97897   FALSE FALSE
## right_eye_center_x                NA       1.0      99.78690   FALSE FALSE
## right_eyebrow_inner_end_x         NA       1.0      32.00739   FALSE FALSE
## mouth_center_bottom_lip_y         NA       1.0      99.36070   FALSE FALSE
## mouth_center_top_lip_x            NA       1.0      32.12104   FALSE FALSE
## mouth_right_corner_y              NA       1.0      31.96477   FALSE FALSE
## nose_tip_y                        NA       1.0      99.84373   FALSE FALSE
## mouth_center_top_lip_y            NA       1.0      32.10683   FALSE FALSE
## left_eye_outer_corner_y           NA       1.0      31.92215   FALSE FALSE
## .pos                              NA       1.0     100.00000   FALSE FALSE
## left_eye_center_y                 NA       1.0      99.73008   FALSE FALSE
## left_eye_inner_corner_y           NA       1.0      31.97897   FALSE FALSE
## mouth_right_corner_x              NA       1.0      31.97897   FALSE FALSE
## .rnorm                            NA       1.0     100.00000   FALSE FALSE
## right_eye_outer_corner_x          NA       1.0      32.02159   FALSE FALSE
## left_eyebrow_outer_end_y          NA       1.5      31.36809   FALSE FALSE
## right_eye_outer_corner_y          NA       1.0      32.02159   FALSE FALSE
## right_eye_inner_corner_y          NA       1.0      32.00739   FALSE FALSE
## left_eyebrow_inner_end_y          NA       1.0      32.05001   FALSE FALSE
## right_eyebrow_outer_end_x         NA       1.0      31.59540   FALSE FALSE
## right_eye_center_y                NA       1.0      99.61642   FALSE FALSE
## right_eyebrow_outer_end_y         NA       1.0      31.60960   FALSE FALSE
## right_eyebrow_inner_end_y         NA       1.0      31.96477   FALSE FALSE
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
```

```
## Warning in myplot_scatter(plt_feats_df, "percentUnique", "freqRatio",
## colorcol_name = "nzv", : converting nzv to class:factor
```

```
## Warning in min(x): no non-missing arguments to min; returning Inf
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in min(diff(sort(x))): no non-missing arguments to min; returning
## Inf
```

```
## Warning in min(x): no non-missing arguments to min; returning Inf
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in stats::runif(length(x), -amount, amount): NAs produced
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
## [1] cor.y            exclude.as.feat  cor.y.abs        cor.high.X      
## [5] freqRatio        percentUnique    zeroVar          nzv             
## [9] is.cor.y.abs.low
## <0 rows> (or 0-length row.names)
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
##             Image           ImageId Image.pxl.1.dgt.1              .lcn 
##                 0                 0                 0              1783
```

```
## [1] "glb_feats_df:"
```

```
## [1] 31 12
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
## 14 select.features          7          0           0 84.629 86.922   2.293
## 15      fit.models          8          0           0 86.922     NA      NA
```

## Step `8.0: fit models`

```r
fit.models_0_chunk_df <- myadd_chunk(NULL, "fit.models_0_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_0_bgn          1          0       setup 87.427  NA      NA
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
## 1               fit.models_0_bgn          1          0       setup 87.427
## 2 fit.models_0_Max.cor.Y.rcv.*X*          1          1      glmnet 87.461
##     end elapsed
## 1 87.46   0.033
## 2    NA      NA
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
## Fitting alpha = 0.1, lambda = 0.00355 on full training set
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
## 66.0759427498  0.0000873936 -0.0829823551 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                           id       feats max.nTuningRuns
## 1 Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      0.817                 0.012  0.003587262
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     3.270851      0.003192721  0.003172451     3.843935       0.00216657
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
## Fitting cp = 0.00979 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = "Max.cor.Y", : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
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
##   n= 5054 
## 
##            CP nsplit rel error
## 1 0.009785893      0         1
## 
## Node number 1: 5054 observations
##   mean=66.38307, MSE=10.73698 
## 
## n= 5054 
## 
## node), split, n, deviance, yval
##       * denotes terminal node
## 
## 1) root 5054 54264.71 66.38307 *
##                     id       feats max.nTuningRuns
## 1 Max.cor.Y##rcv#rpart .pos,.rnorm               5
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      1.873                 0.039            0
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1      3.26607               NA            0     3.850047               NA
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.006771026      0.1773655        0.001570168
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
##                            label step_major step_minor label_minor    bgn
## 2 fit.models_0_Max.cor.Y.rcv.*X*          1          1      glmnet 87.461
## 3         fit.models_0_Low.cor.X          1          2      glmnet 92.441
##     end elapsed
## 2 92.44   4.979
## 3    NA      NA
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
## Fitting alpha = 0.1, lambda = 0.00355 on full training set
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
## 66.0759427498  0.0000873936 -0.0829823551 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                      id       feats max.nTuningRuns
## 1 Low.cor.X##rcv#glmnet .pos,.rnorm              20
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      2.144                 0.007  0.003587262
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     3.268553      0.003192721  0.003172451     3.843935       0.00216657
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.003751506      0.1819376        0.002729321
```

```r
fit.models_0_chunk_df <- 
    myadd_chunk(fit.models_0_chunk_df, "fit.models_0_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                    label step_major step_minor label_minor    bgn    end
## 3 fit.models_0_Low.cor.X          1          2      glmnet 92.441 96.108
## 4       fit.models_0_end          1          3    teardown 96.109     NA
##   elapsed
## 3   3.667
## 4      NA
```

```r
rm(ret_lst)

glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor    bgn    end elapsed
## 15 fit.models          8          0           0 86.922 96.118   9.197
## 16 fit.models          8          1           1 96.119     NA      NA
```


```r
fit.models_1_chunk_df <- myadd_chunk(NULL, "fit.models_1_bgn", label.minor="setup")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_1_bgn          1          0       setup 98.152  NA      NA
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
##                label step_major step_minor label_minor    bgn    end
## 1   fit.models_1_bgn          1          0       setup 98.152 98.163
## 2 fit.models_1_All.X          1          1       setup 98.164     NA
##   elapsed
## 1   0.011
## 2      NA
##                label step_major step_minor label_minor    bgn   end
## 2 fit.models_1_All.X          1          1       setup 98.164 98.17
## 3 fit.models_1_All.X          1          2      glmnet 98.171    NA
##   elapsed
## 2   0.006
## 3      NA
## [1] "fitting model: All.X##rcv#glmnet"
## [1] "    indep_vars: .pos,.rnorm"
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.1, lambda = 0.00355 on full training set
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
## 66.0759427498  0.0000873936 -0.0829823551 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                  id       feats max.nTuningRuns min.elapsedtime.everything
## 1 All.X##rcv#glmnet .pos,.rnorm              20                      2.162
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit max.Adj.R.sq.fit
## 1                 0.006  0.003587262     3.268553      0.003192721
##   max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB max.Rsquared.fit
## 1  0.003172451     3.843935       0.00216657      0.003751506
##   min.RMSESD.fit max.RsquaredSD.fit
## 1      0.1819376        0.002729321
##                label step_major step_minor label_minor     bgn    end
## 3 fit.models_1_All.X          1          2      glmnet  98.171 101.86
## 4 fit.models_1_All.X          1          3         glm 101.861     NA
##   elapsed
## 3   3.689
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
##     Min       1Q   Median       3Q      Max  
## -31.046   -1.243    0.134    1.638   18.492  
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  6.607e+01  9.190e-02 718.969   <2e-16 ***
## .pos         8.789e-05  2.258e-05   3.893   0.0001 ***
## .rnorm      -8.382e-02  4.593e-02  -1.825   0.0680 .  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for gaussian family taken to be 10.70482)
## 
##     Null deviance: 54265  on 5053  degrees of freedom
## Residual deviance: 54070  on 5051  degrees of freedom
## AIC: 26329
## 
## Number of Fisher Scoring iterations: 2
## 
##               id       feats max.nTuningRuns min.elapsedtime.everything
## 1 All.X##rcv#glm .pos,.rnorm               1                      1.112
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit min.aic.fit
## 1                  0.01   0.00358742      3.26856    26329.12
##   max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1      0.003192879  0.003175106      3.84393      0.002169228
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.003753773       0.181952        0.002728187
```

```r
# Check if other preProcess methods improve model performance
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_preProc", major.inc = FALSE,
                label.minor = "preProc")
```

```
##                  label step_major step_minor label_minor     bgn     end
## 4   fit.models_1_All.X          1          3         glm 101.861 105.543
## 5 fit.models_1_preProc          1          4     preProc 105.543      NA
##   elapsed
## 4   3.682
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
## Max.cor.Y.rcv.1X1###glmnet               0                      0.817
## Max.cor.Y##rcv#rpart                     5                      1.873
## Low.cor.X##rcv#glmnet                   20                      2.144
## All.X##rcv#glmnet                       20                      2.162
## All.X##rcv#glm                           1                      1.112
##                            min.elapsedtime.final max.R.sq.fit min.RMSE.fit
## Max.cor.Y.rcv.1X1###glmnet                 0.012  0.003587262     3.270851
## Max.cor.Y##rcv#rpart                       0.039  0.000000000     3.266070
## Low.cor.X##rcv#glmnet                      0.007  0.003587262     3.268553
## All.X##rcv#glmnet                          0.006  0.003587262     3.268553
## All.X##rcv#glm                             0.010  0.003587420     3.268560
##                            max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB
## Max.cor.Y.rcv.1X1###glmnet      0.003192721  0.003172451     3.843935
## Max.cor.Y##rcv#rpart                     NA  0.000000000     3.850047
## Low.cor.X##rcv#glmnet           0.003192721  0.003172451     3.843935
## All.X##rcv#glmnet               0.003192721  0.003172451     3.843935
## All.X##rcv#glm                  0.003192879  0.003175106     3.843930
##                            max.Adj.R.sq.OOB max.Rsquared.fit
## Max.cor.Y.rcv.1X1###glmnet      0.002166570               NA
## Max.cor.Y##rcv#rpart                     NA      0.006771026
## Low.cor.X##rcv#glmnet           0.002166570      0.003751506
## All.X##rcv#glmnet               0.002166570      0.003751506
## All.X##rcv#glm                  0.002169228      0.003753773
##                            min.RMSESD.fit max.RsquaredSD.fit min.aic.fit
## Max.cor.Y.rcv.1X1###glmnet             NA                 NA          NA
## Max.cor.Y##rcv#rpart            0.1773655        0.001570168          NA
## Low.cor.X##rcv#glmnet           0.1819376        0.002729321          NA
## All.X##rcv#glmnet               0.1819376        0.002729321          NA
## All.X##rcv#glm                  0.1819520        0.002728187    26329.12
```

```r
rm(ret_lst)
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                  label step_major step_minor label_minor     bgn     end
## 5 fit.models_1_preProc          1          4     preProc 105.543 105.605
## 6     fit.models_1_end          1          5    teardown 105.606      NA
##   elapsed
## 5   0.062
## 6      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor     bgn     end elapsed
## 16 fit.models          8          1           1  96.119 105.615   9.496
## 17 fit.models          8          2           2 105.616      NA      NA
```


```r
fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor     bgn end elapsed
## 1 fit.models_2_bgn          1          0       setup 107.558  NA      NA
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
## Max.cor.Y.rcv.1X1###glmnet               0  0.003587262      0.003192721
## Max.cor.Y##rcv#rpart                     5  0.000000000               NA
## Low.cor.X##rcv#glmnet                   20  0.003587262      0.003192721
## All.X##rcv#glmnet                       20  0.003587262      0.003192721
## All.X##rcv#glm                           1  0.003587420      0.003192879
##                            max.R.sq.OOB max.Adj.R.sq.OOB max.Rsquared.fit
## Max.cor.Y.rcv.1X1###glmnet  0.003172451      0.002166570               NA
## Max.cor.Y##rcv#rpart        0.000000000               NA      0.006771026
## Low.cor.X##rcv#glmnet       0.003172451      0.002166570      0.003751506
## All.X##rcv#glmnet           0.003172451      0.002166570      0.003751506
## All.X##rcv#glm              0.003175106      0.002169228      0.003753773
##                            inv.elapsedtime.everything
## Max.cor.Y.rcv.1X1###glmnet                  1.2239902
## Max.cor.Y##rcv#rpart                        0.5339028
## Low.cor.X##rcv#glmnet                       0.4664179
## All.X##rcv#glmnet                           0.4625347
## All.X##rcv#glm                              0.8992806
##                            inv.elapsedtime.final inv.RMSE.fit inv.RMSE.OOB
## Max.cor.Y.rcv.1X1###glmnet              83.33333    0.3057308    0.2601501
## Max.cor.Y##rcv#rpart                    25.64103    0.3061784    0.2597371
## Low.cor.X##rcv#glmnet                  142.85714    0.3059458    0.2601501
## All.X##rcv#glmnet                      166.66667    0.3059458    0.2601501
## All.X##rcv#glm                         100.00000    0.3059451    0.2601504
##                             inv.aic.fit
## Max.cor.Y.rcv.1X1###glmnet           NA
## Max.cor.Y##rcv#rpart                 NA
## Low.cor.X##rcv#glmnet                NA
## All.X##rcv#glmnet                    NA
## All.X##rcv#glm             3.798076e-05
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
## 5             All.X##rcv#glm     3.843930  0.003175106      0.003192879
## 3      Low.cor.X##rcv#glmnet     3.843935  0.003172451      0.003192721
## 4          All.X##rcv#glmnet     3.843935  0.003172451      0.003192721
## 1 Max.cor.Y.rcv.1X1###glmnet     3.843935  0.003172451      0.003192721
## 2       Max.cor.Y##rcv#rpart     3.850047  0.000000000               NA
##   min.RMSE.fit
## 5     3.268560
## 3     3.268553
## 4     3.268553
## 1     3.270851
## 2     3.266070
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
## <environment: 0x7fca2cb55038>
```

```r
print(sprintf("Best model id: %s", dsp_models_df[1, "id"]))
```

```
## [1] "Best model id: All.X##rcv#glm"
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
## 66.0759427498  0.0000873936 -0.0829823551 
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
##        All.X..rcv.glmnet.imp imp
## .pos                     100 100
## .rnorm                     0   0
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
## 1862          36.16566          39.99390          23.196134
## 6493          36.94935          44.27019          34.291798
## 4264          94.68928          68.18947          85.039381
## 2534          39.45509          41.69140           9.641790
##      right_eye_center_y left_eye_inner_corner_x left_eye_inner_corner_y
## 1908           56.40497                19.06495                56.29124
## 1862           43.81474                34.14098                40.55803
## 6493           43.88362                      NA                      NA
## 4264           68.47329                      NA                      NA
## 2534           39.64538                      NA                      NA
##      left_eye_outer_corner_x left_eye_outer_corner_y
## 1908                27.57188                56.38438
## 1862                40.60971                38.05205
## 6493                      NA                      NA
## 4264                      NA                      NA
## 2534                      NA                      NA
##      right_eye_inner_corner_x right_eye_inner_corner_y
## 1908                 5.751046                 56.74390
## 1862                25.041023                 43.77863
## 6493                       NA                       NA
## 4264                       NA                       NA
## 2534                       NA                       NA
##      right_eye_outer_corner_x right_eye_outer_corner_y
## 1908                       NA                       NA
## 1862                 22.04996                  43.9963
## 6493                       NA                       NA
## 4264                       NA                       NA
## 2534                       NA                       NA
##      left_eyebrow_inner_end_x left_eyebrow_inner_end_y
## 1908                 17.88872                 51.04292
## 1862                 24.54988                 37.64245
## 6493                       NA                       NA
## 4264                       NA                       NA
## 2534                       NA                       NA
##      left_eyebrow_outer_end_x left_eyebrow_outer_end_y
## 1908                 32.20643                 49.89555
## 1862                 43.18563                 32.53290
## 6493                       NA                       NA
## 4264                       NA                       NA
## 2534                       NA                       NA
##      right_eyebrow_inner_end_x right_eyebrow_inner_end_y
## 1908                  6.921014                  51.24334
## 1862                 22.136163                  42.76767
## 6493                        NA                        NA
## 4264                        NA                        NA
## 2534                        NA                        NA
##      right_eyebrow_outer_end_x right_eyebrow_outer_end_y nose_tip_x
## 1908                        NA                        NA   12.94470
## 1862                   18.1551                   40.9182   21.35837
## 6493                        NA                        NA   35.40312
## 4264                        NA                        NA   89.43859
## 2534                        NA                        NA   24.25615
##      nose_tip_y mouth_left_corner_x mouth_left_corner_y
## 1908   60.06969            22.92336            73.48394
## 1862   66.08490            37.14547            78.11135
## 6493   46.20297                  NA                  NA
## 4264   75.28499                  NA                  NA
## 2534   64.19752                  NA                  NA
##      mouth_right_corner_x mouth_right_corner_y mouth_center_top_lip_x
## 1908             2.245766             74.12832               12.60517
## 1862                   NA                   NA               29.67576
## 6493                   NA                   NA                     NA
## 4264                   NA                   NA                     NA
## 2534                   NA                   NA                     NA
##      mouth_center_top_lip_y mouth_center_bottom_lip_x
## 1908               71.88882                  12.53648
## 1862               78.16098                  33.16245
## 6493                     NA                  35.40312
## 4264                     NA                  89.43859
## 2534                     NA                  24.54844
##      mouth_center_bottom_lip_y
## 1908                  76.62886
## 1862                  82.33012
## 6493                  47.21769
## 4264                  78.54893
## 2534                  71.79694
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          Image
## 1908 5 10 20 29 39 48 61 73 94 112 120 123 127 124 122 123 122 123 123 120 123 126 124 125 127 125 128 132 134 134 138 141 144 147 147 144 151 154 152 155 162 164 167 175 180 184 187 190 192 192 192 192 194 196 197 197 198 200 201 204 205 208 213 217 213 215 218 220 217 217 218 220 216 217 193 169 152 178 169 141 156 160 160 148 129 144 136 157 216 244 246 243 244 243 243 243 12 16 21 35 45 51 63 82 104 118 123 124 128 126 121 125 126 123 128 125 122 125 124 127 128 125 128 133 138 139 143 144 145 149 152 153 155 159 159 160 164 169 176 182 181 182 189 191 191 194 192 193 195 197 198 195 197 198 200 207 209 212 216 213 215 221 219 216 218 220 222 223 217 221 207 181 159 162 191 167 134 118 149 164 159 149 138 130 203 250 243 245 244 243 243 243 17 17 27 42 48 55 67 86 107 118 123 126 127 125 122 123 124 123 127 126 126 125 125 128 127 127 131 135 138 140 143 143 144 151 153 158 160 163 163 161 165 173 181 184 180 179 184 190 194 194 193 195 195 194 195 192 199 201 196 205 210 215 220 217 218 213 215 220 223 223 222 222 220 225 216 199 183 168 187 197 149 131 111 124 107 136 146 120 168 247 245 245 244 243 243 243 12 27 36 41 52 62 74 94 112 122 127 130 125 121 124 123 122 123 125 125 126 124 123 124 124 126 129 130 132 140 144 148 151 154 154 160 167 167 162 164 174 179 182 182 180 178 181 190 198 195 192 191 193 196 196 192 197 196 193 202 214 217 217 223 222 213 219 225 225 222 224 222 218 223 221 213 203 186 182 199 180 155 127 118 100 63 99 124 158 234 248 243 244 244 244 243 22 34 38 55 63 68 79 98 111 121 128 132 124 120 124 126 123 123 126 124 123 125 128 125 128 131 130 130 133 140 142 144 145 151 154 159 166 170 165 166 176 179 185 181 178 176 184 193 197 195 191 189 192 193 192 196 196 197 201 204 219 220 219 222 222 224 221 224 225 224 225 224 220 223 223 212 209 197 185 184 184 167 144 98 89 78 49 89 156 229 247 245 245 245 245 244 25 32 52 56 66 73 84 100 112 122 128 129 126 124 125 127 126 122 122 124 121 123 126 127 129 130 129 130 133 133 142 145 149 151 155 160 165 168 166 168 173 175 187 184 180 180 184 190 190 188 190 191 196 198 198 202 199 201 203 204 214 224 228 228 228 228 224 224 224 222 224 222 220 220 222 213 200 202 192 190 191 182 153 122 75 66 54 51 132 232 248 245 245 245 245 244 33 39 55 55 61 75 88 98 115 125 129 127 126 126 124 125 126 122 121 123 120 123 123 125 126 129 131 131 131 133 140 140 149 151 152 157 164 167 166 164 169 172 182 182 183 177 170 178 185 188 186 192 199 199 200 198 199 201 201 202 209 219 223 228 227 224 227 228 223 222 225 223 220 218 222 219 211 200 194 187 187 190 169 140 114 73 52 47 138 242 246 246 245 245 245 244 46 56 55 57 66 71 85 99 114 124 128 126 126 126 122 125 125 122 122 122 120 125 125 127 125 127 130 128 127 135 139 139 152 156 153 159 167 170 169 167 174 179 183 184 185 179 177 180 184 190 191 190 197 202 200 200 196 200 200 203 209 213 220 223 223 223 220 221 225 225 224 223 221 216 220 220 221 209 195 197 181 173 176 162 129 100 72 44 144 249 246 247 245 245 245 244 50 53 53 59 66 73 88 107 110 118 126 124 122 124 125 127 123 122 122 122 122 123 124 124 127 128 123 126 128 134 135 138 147 152 152 159 169 171 170 170 175 182 184 189 188 181 179 184 180 186 190 193 197 198 196 195 201 205 201 202 209 213 221 226 224 224 224 221 225 223 222 224 223 218 219 218 220 207 200 196 186 168 157 146 152 105 85 74 180 253 243 246 245 244 244 245 45 54 60 59 73 83 93 103 106 117 121 123 123 123 122 122 121 122 121 121 122 122 121 124 124 126 125 129 132 132 135 140 147 152 156 159 169 170 171 172 176 177 180 189 189 182 176 179 178 180 183 191 196 196 198 194 196 199 204 202 208 213 212 216 220 224 226 222 225 224 222 223 220 218 220 221 220 220 205 193 187 180 158 122 125 125 70 101 216 247 246 245 245 244 244 245 56 55 56 75 81 85 89 96 108 119 122 123 121 119 119 124 122 120 121 118 119 123 125 119 116 124 126 128 133 136 140 141 147 152 153 164 166 168 171 170 173 174 178 187 189 184 178 178 177 179 182 188 191 191 196 193 197 202 203 202 206 213 213 215 221 221 226 224 224 224 224 225 218 215 218 222 224 225 214 196 179 173 167 136 92 95 99 139 243 246 246 244 245 244 244 245 53 57 66 78 80 81 89 104 113 118 122 118 118 120 123 122 120 121 124 123 119 118 121 119 116 124 125 125 129 131 136 141 145 148 155 161 162 169 169 165 168 174 178 185 187 182 179 179 174 183 185 186 186 186 188 188 192 194 196 199 196 199 209 214 219 219 223 223 221 220 221 223 216 215 219 223 220 220 211 204 186 167 153 134 111 75 76 147 245 246 245 245 245 244 244 245 61 71 72 73 80 86 96 110 117 119 121 120 123 123 122 119 123 121 122 120 118 120 119 119 121 120 125 128 126 132 134 137 145 151 156 161 162 166 172 171 168 166 173 183 185 181 176 177 173 176 179 183 183 182 184 188 195 194 195 200 199 200 208 213 213 215 219 220 218 223 223 219 216 215 217 222 221 218 211 202 178 164 148 124 117 86 54 152 250 246 246 245 245 245 245 245 66 72 69 75 86 93 101 109 114 119 123 123 121 120 120 119 122 119 118 119 119 122 122 121 121 119 122 125 124 129 127 128 138 146 147 153 159 159 165 165 162 163 170 176 178 175 170 172 172 174 185 188 194 190 191 193 193 200 200 202 198 202 213 218 218 218 218 216 217 221 219 215 215 218 221 222 221 221 212 199 185 159 138 119 107 89 59 156 252 245 246 245 245 245 245 245 70 73 76 83 82 88 100 114 119 122 122 119 117 118 117 118 118 116 119 123 117 114 119 119 117 117 118 121 120 120 127 129 134 141 144 151 162 166 168 169 169 171 173 177 178 179 178 177 177 174 187 194 186 179 191 198 196 198 202 203 196 201 211 216 217 216 215 214 215 220 218 217 216 218 221 220 222 224 215 199 185 165 139 120 101 83 70 176 253 245 246 245 245 245 245 245 70 74 77 73 77 90 104 114 119 122 123 121 123 121 117 116 115 117 120 119 116 115 117 118 120 123 122 124 124 130 136 137 143 143 145 151 160 170 171 171 171 171 172 180 180 179 181 180 183 176 187 194 187 177 190 195 199 199 199 197 199 205 211 219 219 217 216 216 215 220 220 216 215 216 220 222 224 225 212 212 194 164 139 122 97 83 87 201 251 246 245 245 245 245 245 245 68 68 71 73 85 97 103 111 116 122 123 118 116 117 115 115 116 116 116 117 118 119 116 120 121 121 120 122 125 125 129 135 142 141 147 153 156 165 170 171 172 169 176 183 181 182 183 182 179 183 187 187 189 191 191 189 192 196 198 199 203 209 212 218 220 216 214 213 214 220 220 217 215 217 221 218 219 223 218 218 205 174 136 116 106 78 82 210 250 245 245 245 244 244 244 244 59 67 78 78 85 93 101 111 117 120 121 118 113 119 117 113 116 116 115 116 120 122 115 114 117 122 120 119 124 126 128 134 139 143 153 157 158 163 169 170 174 172 179 182 182 183 178 179 175 186 187 172 189 187 187 186 192 195 198 199 201 207 212 218 221 217 216 213 213 217 218 217 216 217 221 221 224 221 220 221 202 181 153 113 91 70 74 212 250 245 245 245 245 245 245 245 67 76 79 80 83 92 104 115 119 119 121 120 118 116 115 116 115 115 115 114 116 119 117 115 117 119 119 121 126 127 128 134 140 146 155 157 158 165 168 168 174 177 178 179 183 184 179 172 176 191 187 160 181 189 184 182 190 192 198 202 204 214 215 218 221 217 219 221 219 220 219 216 214 216 220 221 219 222 221 216 207 184 154 129 94 60 98 221 249 247 246 246 245 245 245 245 85 83 76 80 85 95 108 114 116 119 122 123 121 115 115 116 116 119 120 119 119 118 121 119 120 116 118 120 125 125 129 137 142 147 153 154 162 166 170 172 177 179 179 181 183 183 184 173 175 182 183 185 189 189 189 187 192 195 202 214 211 217 218 223 224 223 225 224 221 223 222 219 216 217 219 220 214 220 221 217 217 194 149 123 119 70 95 225 249 247 246 246 245 245 245 245 80 76 82 84 87 93 104 112 116 120 121 120 121 120 118 115 115 115 121 121 117 117 121 119 120 121 121 122 125 126 129 135 139 148 152 151 163 168 175 175 177 181 183 187 188 183 180 175 169 175 179 182 189 189 191 192 191 193 201 212 212 219 220 222 227 228 226 223 224 225 222 223 220 215 220 222 218 218 218 218 219 205 164 119 104 93 90 209 250 246 246 245 245 245 245 244 76 79 83 87 88 95 105 116 122 124 121 120 120 121 117 117 119 118 122 121 118 122 125 125 125 126 125 126 129 131 133 138 141 150 158 154 156 166 173 171 172 182 186 193 194 184 182 178 168 171 177 184 193 190 191 193 197 198 204 211 215 221 221 221 227 227 225 224 223 227 228 226 223 217 215 217 222 215 211 215 216 215 177 124 91 94 123 214 250 245 246 246 245 245 245 244 90 83 77 85 95 106 113 117 117 118 120 121 119 120 118 120 120 120 122 123 122 124 128 127 128 130 127 128 132 134 136 139 140 146 158 161 164 166 173 177 175 180 185 193 196 182 179 172 165 169 176 188 192 194 193 191 199 200 203 210 215 216 219 221 224 221 217 210 203 204 205 212 220 218 217 218 221 218 214 213 215 221 196 139 106 86 106 214 248 243 245 246 245 245 245 244 84 82 85 94 103 112 117 119 118 118 123 125 123 123 122 122 122 121 122 125 126 125 129 130 129 131 131 130 129 132 136 138 139 141 147 158 171 169 171 181 178 183 185 191 197 184 181 168 160 169 179 181 189 194 194 193 196 196 202 207 214 219 213 210 213 214 212 200 191 183 180 183 193 207 216 219 217 218 215 215 218 217 212 173 124 90 85 184 242 247 246 244 245 245 245 244 86 94 103 109 111 117 120 123 123 122 124 124 123 125 123 125 125 125 123 125 127 127 132 130 130 132 133 133 131 133 137 141 139 143 147 156 164 170 173 177 178 185 188 188 194 178 176 172 163 169 178 183 191 189 189 194 196 200 207 207 205 207 211 208 206 198 185 177 175 177 182 180 173 187 185 194 215 220 219 218 217 219 217 185 142 95 84 140 207 237 245 244 245 245 245 245 100 106 116 121 121 122 124 123 121 123 124 123 121 124 124 126 127 128 127 129 129 132 130 127 132 136 134 138 137 137 138 139 141 145 147 150 158 166 176 174 180 185 187 193 202 179 171 173 161 164 174 182 190 188 188 192 198 200 203 207 204 195 186 182 179 171 160 153 148 158 170 176 175 178 175 176 198 215 216 215 216 215 215 194 151 108 103 150 199 228 245 245 245 245 245 245 112 123 126 130 128 127 128 125 123 123 124 123 124 125 128 127 129 129 124 118 116 125 121 124 126 132 134 134 133 136 139 138 147 151 149 151 158 163 171 170 176 185 185 193 204 171 159 169 159 154 180 182 182 189 190 193 197 195 192 187 166 151 142 136 137 131 132 142 146 153 162 168 176 179 184 181 182 192 208 216 214 213 218 203 155 120 110 159 209 236 246 247 246 245 244 244 118 128 133 135 127 129 131 128 125 123 123 124 126 126 131 127 126 123 113 103 107 115 113 109 110 111 117 122 129 133 134 135 141 151 151 151 155 158 169 169 170 179 183 193 206 169 149 166 161 144 170 183 178 187 187 184 179 166 146 129 101 112 120 115 99 87 110 131 144 145 155 156 151 148 146 147 166 178 200 213 213 214 216 210 165 120 105 132 191 242 245 245 245 244 244 244 123 136 139 136 131 132 133 127 126 123 123 126 125 127 132 127 114 103 93 89 86 83 86 84 96 96 99 100 104 112 120 130 133 145 148 150 151 157 162 161 164 173 179 187 200 161 135 156 166 154 161 176 182 177 170 162 151 132 112 96 88 91 83 78 82 88 105 132 151 147 147 143 142 142 141 151 158 173 195 205 213 219 218 215 177 116 99 131 202 245 247 245 244 243 244 245 127 141 142 140 137 132 132 128 127 124 122 126 128 124 112 97 88 77 78 74 69 78 77 77 79 79 70 61 64 76 82 86 91 101 124 138 142 149 154 155 157 167 177 181 200 154 126 153 166 149 155 164 170 171 145 141 122 108 113 99 91 81 79 86 103 118 141 163 166 177 178 171 170 163 169 183 179 187 192 198 215 219 220 213 179 122 110 152 219 248 245 244 244 243 243 243 131 144 145 143 141 134 130 127 124 121 121 122 113 96 83 76 69 68 72 72 77 72 70 55 48 42 31 28 33 39 46 50 63 70 84 108 129 131 146 149 153 160 172 183 196 158 126 137 164 139 141 156 153 148 148 131 110 112 106 100 98 104 112 132 156 167 173 170 177 185 199 198 196 195 191 190 190 185 179 192 214 218 218 213 177 133 118 163 227 247 244 244 244 243 243 243 133 148 149 144 140 136 131 129 126 125 124 118 91 72 64 59 53 46 48 43 47 49 55 47 34 25 26 27 24 24 32 34 38 47 61 68 97 105 115 133 142 147 163 179 186 150 128 135 154 152 142 152 152 146 138 133 126 122 127 127 129 135 141 154 165 170 175 176 181 181 190 194 192 197 198 187 187 184 185 204 218 217 219 216 182 128 120 174 231 246 244 244 244 243 243 243 133 146 149 145 141 137 129 129 124 122 114 95 76 67 66 59 48 46 43 38 49 66 69 67 57 49 42 40 38 34 29 32 33 32 45 54 68 85 102 116 127 141 155 175 179 138 127 142 146 152 141 143 152 149 142 141 143 146 142 129 132 138 139 153 158 165 168 170 172 176 181 186 191 196 195 195 188 189 200 210 217 219 218 216 199 132 124 177 228 246 245 244 244 243 243 243 129 143 149 145 142 138 131 127 116 116 109 95 90 84 69 60 63 62 62 76 88 89 93 103 108 104 102 94 78 66 50 45 48 50 49 60 70 75 85 103 116 129 142 163 162 134 139 147 149 153 146 141 150 144 148 151 145 131 118 112 107 102 99 100 99 102 109 117 120 124 133 153 172 183 193 191 193 198 201 214 220 220 219 217 200 140 143 182 228 246 245 244 243 243 243 243 122 140 150 146 142 136 134 125 118 121 119 103 92 89 87 84 80 76 80 101 112 107 108 115 116 118 123 123 118 107 93 76 62 57 63 67 71 74 76 86 97 114 127 140 145 138 139 156 159 167 168 158 157 157 144 124 113 108 95 74 51 29 26 32 32 34 49 52 47 71 80 85 126 175 188 191 197 203 207 216 221 222 218 216 204 147 167 197 233 245 243 245 243 243 243 243 122 139 149 146 144 137 133 118 106 104 106 102 95 85 90 99 95 90 92 100 103 108 109 111 115 116 123 125 128 129 119 103 83 65 54 50 59 62 68 79 87 98 114 128 131 138 145 168 178 190 190 184 180 156 115 103 100 89 69 60 49 55 35 60 148 84 112 149 111 74 87 96 84 144 195 201 202 209 217 219 215 218 220 216 200 152 177 195 231 247 244 243 243 243 243 243 121 134 147 142 141 137 133 123 116 107 93 92 94 91 91 96 97 94 95 99 94 96 98 98 104 102 100 96 79 65 60 52 57 60 59 54 47 45 54 69 76 88 102 123 131 141 156 174 194 212 221 211 184 134 97 93 87 66 81 112 97 115 78 31 97 104 106 195 193 130 107 108 111 128 166 197 205 214 223 224 219 220 221 215 195 163 184 199 234 247 243 243 243 243 243 243 108 129 142 143 140 137 130 126 120 113 100 98 99 94 89 102 104 99 95 93 88 80 68 51 40 29 22 11 57 74 9 43 33 27 41 58 56 39 41 51 63 77 92 115 128 140 160 183 208 223 233 224 181 129 105 111 87 77 107 127 108 115 128 109 119 102 149 215 183 164 183 184 176 178 180 194 212 223 224 227 226 225 219 215 199 173 185 210 238 246 244 244 243 243 243 243 108 131 142 143 141 136 130 125 107 100 104 102 94 85 93 99 104 99 84 69 58 36 25 35 30 44 50 10 43 93 63 113 126 69 33 42 63 67 40 48 61 74 89 112 127 140 156 182 213 224 233 226 205 184 176 165 96 91 102 117 130 130 145 154 160 153 184 191 189 219 217 205 215 212 203 207 214 221 227 228 225 223 221 219 202 176 189 218 240 244 244 244 243 243 243 243 110 131 145 144 138 133 128 126 111 98 102 90 85 94 95 101 94 70 55 46 32 21 46 86 75 73 89 64 58 82 75 130 129 122 89 32 55 106 91 57 57 75 90 113 126 138 155 181 201 220 230 225 216 214 203 189 152 116 100 113 127 137 151 159 170 184 188 200 206 206 204 214 221 221 219 215 219 222 227 228 225 222 221 218 201 177 195 225 243 244 243 243 243 243 243 243 121 129 142 142 139 133 129 128 124 120 105 86 94 102 89 86 77 63 52 51 57 76 91 98 101 80 83 85 87 117 107 119 115 111 106 95 86 118 128 94 64 73 89 110 122 136 154 178 192 212 226 228 218 218 181 162 151 115 96 102 112 129 145 151 166 185 190 192 195 204 215 223 224 222 224 223 225 225 227 229 227 224 219 215 202 180 197 227 244 245 245 242 243 243 243 243 124 133 143 144 141 136 133 134 135 128 124 123 119 110 102 89 78 76 87 106 114 111 106 103 110 108 105 110 115 111 121 125 124 113 114 118 122 133 124 104 72 74 92 111 124 135 150 171 189 208 226 226 220 214 201 160 144 120 100 98 111 115 111 126 142 154 169 183 197 203 212 213 219 224 228 229 225 223 225 226 226 223 217 215 201 182 203 229 243 245 245 243 242 243 243 242 133 143 147 146 144 139 136 138 137 132 134 133 125 115 107 90 90 101 114 112 114 116 114 115 114 113 111 111 113 119 124 126 115 101 101 109 121 126 111 100 89 88 102 115 123 130 148 168 186 206 226 224 225 216 212 189 159 151 134 117 123 131 128 131 135 148 167 182 197 206 214 218 223 225 227 229 228 229 228 224 225 222 217 215 198 183 207 231 243 245 244 243 242 243 243 241 149 148 146 146 146 140 134 134 134 131 131 133 130 116 109 111 116 118 120 120 112 106 108 109 112 114 114 116 118 125 124 117 107 103 111 121 126 120 106 107 102 100 110 115 118 124 138 159 183 204 224 226 227 223 217 211 187 174 170 171 173 177 174 173 175 178 184 195 210 215 214 215 218 221 225 228 231 232 228 226 225 223 219 215 196 181 208 232 243 244 243 242 243 243 241 240 156 151 149 147 145 139 134 133 132 132 131 133 132 131 128 127 126 129 130 126 116 110 100 92 93 92 88 89 91 91 91 97 110 115 123 130 126 120 112 114 107 107 113 112 115 123 133 149 179 203 223 226 227 226 223 224 222 196 180 177 177 189 194 195 201 202 202 209 222 229 228 226 221 218 220 222 225 230 229 224 221 222 218 216 196 179 209 233 244 243 242 243 243 241 239 238 164 155 150 149 148 142 139 137 133 132 131 136 137 137 135 133 133 131 127 125 119 111 104 101 92 90 92 95 98 111 122 125 130 131 133 135 128 118 115 115 113 118 118 115 114 125 132 145 178 205 222 225 226 227 224 226 229 222 201 177 173 176 187 193 197 205 214 218 217 224 228 221 220 218 222 225 229 232 230 224 220 221 219 214 196 191 214 235 244 243 242 242 242 241 239 237 170 162 154 151 145 145 142 136 133 131 132 135 136 134 136 137 136 135 132 127 122 120 114 116 117 124 131 131 131 136 137 136 133 135 142 138 120 119 124 120 115 124 125 117 115 123 127 141 171 198 214 221 224 225 225 227 226 231 223 201 188 174 177 186 189 195 208 216 216 217 219 218 219 217 219 227 233 233 230 225 223 220 218 212 210 213 217 237 244 243 242 241 242 241 239 236 175 166 154 152 146 147 143 138 131 132 135 136 137 138 140 140 138 139 140 141 137 137 135 134 131 134 136 132 130 133 132 137 139 140 141 129 120 124 127 125 120 126 126 116 114 120 129 141 165 192 206 216 220 223 223 225 226 229 229 223 207 199 183 183 191 194 197 207 208 211 210 214 219 220 222 230 237 234 230 224 221 218 214 213 219 213 212 239 244 243 242 241 241 240 237 234 175 169 158 154 149 146 141 141 133 131 134 137 135 138 140 139 140 141 141 142 138 138 137 137 134 137 138 133 130 133 133 136 141 140 133 126 127 126 124 123 127 124 118 114 116 122 130 139 165 191 207 217 218 222 223 224 225 228 225 216 216 217 203 195 197 200 199 206 206 213 214 215 219 224 231 236 238 234 230 226 222 217 212 214 221 213 216 239 244 243 242 241 240 239 236 233 173 171 164 153 149 152 146 144 134 134 136 134 136 135 136 141 141 141 141 143 142 143 141 142 139 138 140 137 136 137 136 142 140 134 127 131 132 130 127 123 128 125 120 117 117 126 131 138 162 185 206 217 217 224 227 228 227 230 225 220 221 214 210 210 206 202 201 204 209 214 209 213 225 232 236 238 235 231 227 226 222 217 214 213 222 213 218 241 243 243 242 241 239 238 236 232 172 172 165 155 150 155 150 147 140 134 137 136 138 137 141 141 141 144 141 146 145 146 145 145 143 144 143 143 141 141 141 140 136 134 131 133 133 130 129 126 128 128 125 121 122 131 132 143 155 181 206 215 216 222 229 233 228 230 233 228 225 218 214 211 209 207 206 207 208 209 211 217 226 235 237 233 232 230 226 223 221 216 214 213 219 214 219 241 242 243 243 241 239 236 235 231 169 170 165 157 155 158 152 144 139 134 137 138 137 137 140 140 141 144 142 144 147 144 142 148 145 148 146 139 140 141 138 137 135 135 141 134 131 133 129 122 122 125 121 122 127 131 133 142 155 178 202 219 226 223 230 230 226 219 218 231 228 222 218 214 211 210 207 209 210 210 220 225 227 234 236 234 229 228 226 223 222 218 212 212 218 213 219 240 243 243 241 240 237 234 233 229 165 167 164 158 159 158 156 146 140 135 139 139 138 139 137 137 138 140 142 141 145 141 139 148 147 149 142 137 143 141 139 137 137 138 137 137 138 132 113 108 116 120 123 122 128 137 133 135 155 182 203 228 235 226 226 223 222 219 195 204 218 221 220 219 219 216 212 214 216 217 225 227 229 234 237 233 230 228 227 224 223 218 212 211 216 211 221 241 243 242 240 238 235 232 229 225 162 164 163 160 159 157 160 151 142 137 139 137 138 138 134 133 136 137 140 138 139 140 138 143 146 145 145 146 142 139 142 141 139 140 141 141 137 120 98 109 121 122 121 119 132 137 134 136 155 177 203 233 234 227 226 219 216 223 200 175 194 204 215 219 223 218 213 216 219 222 224 229 230 232 232 231 230 229 225 225 223 216 212 211 212 210 224 242 242 241 239 237 234 230 227 223 162 165 165 160 156 156 159 154 141 140 136 137 137 134 134 135 138 138 137 137 140 143 145 145 145 147 148 143 139 139 141 140 138 139 141 143 132 107 104 118 126 125 122 126 138 135 136 138 153 177 204 234 237 226 228 220 213 218 217 174 160 184 200 204 213 217 212 213 216 220 223 225 227 231 232 230 229 228 223 222 222 218 212 211 210 212 228 243 242 240 238 236 234 229 226 221 162 166 164 158 157 159 162 159 144 135 135 134 134 134 134 136 135 135 136 139 142 144 147 147 145 146 149 144 138 139 139 140 141 141 142 139 118 85 109 129 129 128 131 134 141 136 138 139 152 177 209 233 243 231 229 228 223 221 222 194 144 155 174 189 202 210 212 211 212 216 223 224 226 229 229 227 227 226 224 223 218 217 215 211 210 211 232 241 242 239 237 235 232 228 225 220 155 165 163 158 160 157 160 159 148 135 133 131 133 132 132 136 138 138 137 138 138 139 142 143 144 144 142 142 143 142 139 138 140 141 139 133 89 74 117 134 135 136 134 132 133 128 132 134 141 158 184 212 221 219 223 227 227 224 223 208 160 135 154 170 187 199 210 211 208 210 216 220 223 225 224 224 223 223 223 223 216 216 214 209 202 206 235 241 240 237 236 233 230 226 222 217 146 163 163 158 157 151 156 158 150 140 131 130 133 130 130 132 136 136 133 133 134 136 140 142 141 141 141 139 138 139 140 141 140 138 136 109 61 78 119 129 131 129 126 120 122 117 116 116 118 136 158 186 204 206 211 218 215 211 216 213 184 144 141 156 172 188 200 206 206 208 211 212 218 220 220 223 222 219 220 221 220 216 212 202 188 215 238 242 239 236 235 231 228 224 220 215 135 161 159 157 154 148 153 161 155 141 129 128 130 131 130 133 134 131 131 133 134 136 140 137 139 136 140 139 137 137 139 138 139 135 130 77 49 78 113 116 117 112 104 102 103 108 103 104 105 115 134 166 189 195 170 146 158 181 197 210 207 168 139 150 166 178 189 199 203 205 206 209 217 221 222 224 224 220 219 218 220 213 209 198 205 234 241 240 238 237 234 230 225 221 217 213 124 154 153 156 154 148 148 156 156 141 130 128 127 130 130 131 131 128 129 132 131 130 136 136 134 134 135 134 135 138 141 137 135 134 115 60 49 69 102 97 89 73 47 38 44 71 89 94 92 101 123 156 182 121 52 45 77 132 182 208 212 190 154 142 160 177 187 195 198 201 205 208 216 219 221 222 220 221 217 217 219 214 210 217 233 241 240 238 238 236 232 228 224 220 216 211 112 143 155 156 153 147 144 151 155 143 130 121 124 129 128 126 129 127 127 130 129 129 133 137 133 135 133 130 131 133 135 135 130 133 109 61 61 63 79 84 67 33 17 20 31 47 68 87 84 80 121 181 201 187 143 98 91 151 203 219 222 207 181 152 150 166 184 197 199 202 206 208 215 218 220 216 216 221 218 217 217 214 211 228 240 242 240 239 238 235 230 227 222 218 213 209 102 128 157 152 155 150 144 152 153 146 134 123 123 126 125 124 128 127 128 129 130 133 130 134 133 130 130 131 134 129 127 132 133 138 128 78 66 63 63 63 52 35 36 52 64 64 66 76 78 72 127 173 193 219 214 180 191 212 221 222 223 214 197 173 155 159 174 191 200 203 204 207 211 214 217 218 222 220 215 217 214 211 210 226 237 241 239 239 238 234 229 225 221 216 211 206 91 115 157 151 152 149 145 149 154 147 139 130 125 125 121 124 125 123 124 126 127 131 129 128 132 132 129 129 131 126 130 135 137 140 143 118 100 93 81 66 56 51 53 54 61 60 56 62 72 84 122 167 186 191 199 209 218 220 225 224 221 216 206 191 172 164 175 184 193 204 209 211 213 216 216 213 218 217 216 217 216 210 211 231 239 240 238 237 235 233 229 224 220 215 210 205 68 83 147 155 151 145 144 147 153 147 142 135 128 124 124 123 124 125 123 124 124 128 130 128 130 130 130 130 128 124 131 131 131 142 140 137 134 127 116 105 89 78 71 75 71 76 86 95 90 95 130 162 174 182 192 191 196 218 222 221 221 219 211 195 189 177 176 186 189 200 206 208 211 215 215 212 211 214 216 216 216 207 209 232 240 240 239 236 234 231 229 223 220 214 209 205 108 137 161 147 149 144 143 143 151 147 141 131 129 125 124 120 122 124 123 125 125 127 129 131 131 130 131 130 133 132 133 137 142 145 139 137 136 135 127 119 116 113 97 95 100 102 112 118 125 108 122 143 168 182 172 178 193 211 222 221 222 222 219 197 182 182 187 190 195 200 200 200 204 211 213 211 207 213 217 215 214 208 214 234 239 240 238 236 234 230 227 222 218 213 208 204 245 255 214 147 145 146 145 144 149 146 143 138 131 125 121 123 121 121 123 126 125 126 128 129 130 132 128 128 133 130 131 139 141 140 136 140 136 130 128 115 104 98 91 93 103 114 131 143 153 132 116 139 165 160 152 161 175 181 197 208 219 217 217 214 197 181 185 188 197 200 197 197 203 209 210 212 214 212 216 217 215 207 213 234 239 239 237 235 233 229 225 220 217 211 207 203 255 254 239 163 142 145 146 148 151 144 141 137 132 120 123 122 120 120 122 124 126 125 127 129 126 127 127 127 125 130 133 137 139 136 133 134 129 130 118 101 91 89 86 100 105 128 148 171 159 148 135 132 149 150 150 149 151 151 179 187 191 194 188 194 200 189 185 190 199 205 199 196 201 207 211 211 216 215 217 217 214 209 220 236 238 238 237 235 232 228 224 220 217 211 206 201 253 253 254 184 138 147 145 147 150 143 141 138 133 122 121 122 119 120 121 119 119 121 122 124 126 126 125 128 126 128 134 136 134 135 129 122 117 115 99 89 94 96 86 94 113 129 163 185 182 160 144 140 151 153 150 137 144 152 173 176 172 178 174 176 185 189 193 197 204 213 202 193 198 207 210 208 212 217 217 215 212 207 222 237 238 237 236 234 231 227 224 220 216 211 205 200 254 253 255 211 140 145 145 147 147 142 141 138 130 121 120 123 121 120 118 114 117 121 120 121 124 126 123 120 123 124 134 127 127 120 108 101 93 87 83 80 95 98 91 100 115 140 185 196 193 173 157 159 173 177 165 153 164 165 169 176 163 164 169 185 169 154 180 189 200 213 208 196 197 208 206 206 212 217 213 213 212 209 221 236 238 236 235 234 231 227 224 219 215 210 204 199 253 252 255 236 152 140 146 147 146 142 141 139 132 128 124 124 122 116 112 114 122 119 117 121 121 125 120 120 123 126 126 132 116 103 100 83 73 72 72 89 106 108 112 122 133 153 177 190 195 182 173 173 182 187 190 187 188 187 184 179 156 154 163 164 162 158 164 173 187 212 215 203 193 200 208 208 213 215 212 210 210 207 222 236 237 236 235 233 230 226 223 218 213 208 203 197 252 253 255 251 166 134 146 148 145 143 143 142 135 128 122 127 127 123 119 117 117 117 115 116 121 119 115 122 124 124 124 110 92 80 70 71 74 77 84 101 123 135 131 133 126 118 125 127 147 162 161 153 148 149 161 162 177 197 203 196 176 171 170 156 150 144 161 169 179 207 217 207 197 199 209 211 213 214 213 210 207 206 225 238 237 236 234 233 230 226 223 217 213 208 202 197 255 255 254 255 194 132 138 144 143 142 144 142 139 129 126 129 125 120 117 112 110 116 117 116 119 114 114 120 122 116 101 91 80 75 78 74 89 92 100 121 127 117 99 90 86 83 94 94 97 114 126 127 125 140 145 151 150 146 152 164 171 179 178 158 147 135 154 166 182 199 215 212 200 200 204 209 212 212 211 209 205 207 229 237 236 236 234 231 228 224 220 215 210 207 201 196 255 255 254 255 204 122 129 141 142 142 144 145 141 133 129 127 124 121 119 118 117 113 114 117 117 116 119 122 118 99 84 85 90 89 76 83 100 108 98 92 82 75 78 85 86 84 77 68 72 90 96 96 84 102 106 121 127 119 123 132 127 123 127 107 117 139 143 167 191 203 214 212 202 200 201 207 210 211 210 209 203 210 233 237 236 236 233 231 228 224 220 214 210 206 198 193 255 255 254 255 221 128 127 136 141 142 142 145 144 133 126 123 123 120 119 121 120 114 115 117 115 117 120 122 112 97 87 93 99 87 78 92 93 80 65 56 61 62 59 55 58 60 57 55 60 68 64 78 71 73 91 111 111 118 119 124 95 90 104 77 77 112 124 157 176 201 215 214 205 203 203 204 208 209 209 206 204 220 238 235 235 235 233 230 227 223 219 214 210 206 199 194 255 255 254 255 235 147 119 134 143 141 139 142 146 135 126 124 123 118 117 121 118 118 120 116 116 116 119 121 110 106 91 97 102 97 82 50 37 33 35 38 40 47 54 55 70 80 103 120 132 139 124 136 153 156 164 193 215 225 197 180 177 194 187 153 111 98 115 135 171 197 214 216 210 207 204 202 207 209 209 205 198 218 238 235 235 235 232 230 227 223 219 212 206 203 198 193 255 255 254 255 242 166 130 135 143 142 140 139 145 137 128 123 119 122 120 119 119 115 113 116 118 117 119 119 112 104 102 107 101 95 82 74 69 60 55 61 68 80 93 103 106 112 125 140 146 164 164 158 162 172 199 228 233 217 210 215 220 227 214 193 160 142 136 140 171 198 212 217 212 208 204 204 207 209 204 200 198 216 236 237 236 235 234 231 226 222 219 212 208 202 197 193 255 255 255 255 244 177 140 125 136 143 136 143 146 137 126 120 118 119 115 114 117 113 112 113 116 116 119 118 107 103 119 113 94 98 111 124 130 124 113 109 105 99 99 102 101 106 116 137 151 173 177 157 171 191 203 209 199 204 217 221 217 226 227 210 189 173 162 164 176 199 212 216 212 208 205 203 202 206 200 194 205 225 236 236 236 235 233 230 226 222 218 213 209 202 196 191 255 255 255 255 249 182 143 127 123 128 137 139 141 139 126 117 117 116 113 113 114 111 111 115 116 113 120 123 120 116 111 113 108 108 131 123 126 129 127 131 122 115 112 108 105 106 109 127 140 159 171 164 176 180 183 192 205 215 216 222 222 224 222 219 200 181 178 173 179 200 208 212 211 208 202 200 202 198 195 194 214 234 236 235 234 234 232 229 226 221 217 212 207 200 194 189 255 255 255 255 253 190 151 139 123 124 134 130 134 142 129 115 114 114 115 114 116 115 110 112 115 113 118 126 125 118 110 109 119 125 126 131 131 125 124 128 127 127 131 131 124 120 117 123 127 136 149 165 159 168 180 197 211 210 215 218 223 221 218 214 194 179 182 174 176 203 206 209 209 205 200 202 199 196 192 196 221 235 236 234 233 233 231 228 225 221 217 211 205 199 194 190 254 255 254 252 254 202 148 138 120 120 130 123 130 138 128 116 113 109 110 111 114 116 111 110 114 114 116 120 121 116 122 120 116 127 130 129 128 124 120 121 122 125 127 125 128 123 121 123 122 118 122 148 147 158 178 193 203 208 211 211 215 212 216 211 202 187 171 180 185 198 207 206 209 202 199 201 197 196 189 199 225 234 234 234 233 232 230 227 224 220 215 209 204 198 194 188 255 254 252 253 255 212 145 144 126 113 119 121 122 127 129 122 111 107 106 108 111 113 110 106 111 116 114 115 119 119 118 126 122 123 123 120 126 121 120 121 117 123 120 117 119 122 129 139 133 122 117 135 140 147 167 188 195 205 213 215 209 207 214 216 215 194 172 187 193 195 201 195 198 200 201 199 197 198 193 215 235 235 234 234 232 232 230 226 223 219 214 208 203 198 192 187 252 251 255 252 215 185 155 140 128 115 112 116 118 119 126 119 111 102 105 108 107 114 111 107 110 110 110 114 117 116 111 111 118 116 116 126 126 117 118 116 115 121 114 113 106 110 121 139 132 139 139 129 124 143 150 171 191 194 208 215 204 204 210 206 197 188 182 193 197 196 199 190 187 198 196 196 198 189 197 226 234 232 234 233 231 230 228 225 221 217 213 207 203 197 192 186 252 255 240 157 125 195 155 140 134 119 105 106 114 114 119 118 114 107 105 105 107 111 107 105 110 112 114 115 112 113 112 108 116 120 120 122 123 124 126 125 113 112 115 106 103 108 111 132 118 93 101 110 110 135 138 157 182 190 203 213 208 203 199 203 202 195 185 181 188 194 203 196 194 201 199 199 195 188 215 232 232 230 233 231 230 229 227 224 220 216 212 206 202 196 191 185 255 235 134 40 149 227 160 142 136 122 106 94 103 109 112 119 116 113 102 97 103 105 107 108 108 110 114 109 103 95 99 111 114 114 126 123 116 119 127 127 116 115 115 106 108 111 114 125 104 85 100 100 113 139 147 167 185 200 209 210 210 212 208 208 198 189 192 196 202 203 200 196 196 202 203 197 185 200 229 230 229 229 230 229 229 228 226 223 219 215 210 206 201 196 190 185 239 141 26 59 216 229 171 139 136 122 110 99 95 100 104 109 113 113 104 100 101 100 103 104 103 106 106 104 104 99 107 111 114 119 124 125 122 127 131 123 123 128 123 121 120 123 120 118 116 112 119 121 141 160 169 187 199 204 208 212 218 214 209 207 200 187 188 180 186 203 199 197 195 195 197 192 190 219 232 227 228 229 230 229 229 227 224 221 217 214 210 204 200 194 189 183 159 49 7 132 232 226 180 140 132 124 116 104 95 96 100 103 109 111 107 102 101 99 98 100 102 101 101 103 106 102 103 111 116 120 121 120 124 129 126 129 133 131 133 134 134 136 138 136 135 147 146 151 173 186 199 201 202 208 212 215 223 214 201 191 194 176 175 182 192 201 198 196 196 196 186 189 212 226 230 227 228 228 228 228 228 226 223 220 217 213 209 203 198 193 187 182 67 12 32 183 235 227 188 137 128 121 114 104 95 89 90 95 97 101 103 98 97 96 95 98 98 97 100 104 102 104 101 111 113 113 120 120 126 122 124 135 138 136 138 139 139 148 153 146 152 169 167 167 174 178 192 203 214 211 218 222 222 215 208 193 197 201 189 185 197 197 200 200 193 187 182 204 227 228 229 229 228 228 227 227 227 225 222 219 216 213 209 203 199 194 188 183 26 2 75 215 230 229 203 145 124 118 113 105 95 84 84 85 91 93 102 100 96 93 90 93 96 95 93 100 100 100 101 105 108 115 121 117 118 123 130 128 137 141 141 137 142 143 150 153 161 164 159 157 155 166 179 189 201 209 216 216 212 206 208 198 196 195 187 187 192 196 191 198 186 176 202 224 228 230 230 228 228 227 226 225 225 224 221 219 216 212 208 203 198 193 188 182 17 3 115 223 225 225 212 155 121 114 112 104 95 87 86 79 86 89 99 95 95 94 89 88 87 91 94 96 97 96 90 99 115 118 118 110 109 124 125 122 127 132 134 131 126 115 141 151 160 163 156 151 150 167 184 187 185 195 204 207 217 214 197 178 176 184 193 192 197 194 192 188 177 199 221 227 226 227 227 227 226 226 225 224 224 223 221 219 215 211 207 202 196 191 186 180 11 15 144 222 225 219 217 169 123 112 109 104 97 88 86 83 75 76 86 88 93 92 89 83 82 91 87 84 92 90 88 105 109 113 109 102 106 107 115 117 104 115 115 108 110 115 134 151 168 159 164 166 147 153 165 176 182 184 188 192 196 197 187 168 150 180 200 192 196 194 183 172 186 219 224 226 226 227 227 226 226 225 225 224 223 223 221 219 215 210 207 202 197 191 186 180 3 29 166 220 224 218 217 187 131 108 101 99 96 87 84 82 78 71 75 84 89 88 86 86 84 85 87 81 80 77 82 93 87 96 93 97 102 107 116 98 92 97 95 99 101 114 131 143 157 162 156 144 152 154 152 171 182 172 169 173 162 168 183 178 171 178 183 184 194 184 165 184 210 225 222 224 226 226 226 225 225 225 224 223 222 222 219 218 215 211 207 202 196 190 185 178 2 43 174 218 219 219 216 203 149 110 96 92 94 86 83 80 79 78 74 71 82 86 87 87 81 76 77 79 76 75 74 64 78 105 107 98 82 91 103 97 95 88 85 86 100 124 132 137 142 159 165 151 146 148 157 168 174 164 159 165 165 164 181 186 201 187 169 166 181 163 180 204 180 208 225 223 224 226 225 225 225 224 224 223 222 222 220 218 215 210 206 201 195 190 184 178 1 47 172 213 218 218 215 208 173 114 98 91 92 89 86 79 75 76 78 67 69 74 86 86 75 75 74 75 70 70 67 65 84 101 100 81 75 82 88 95 79 81 88 94 102 104 119 134 149 157 162 149 133 134 148 167 165 173 169 157 156 165 163 166 187 175 154 155 160 183 207 198 125 159 224 226 222 227 224 226 225 224 223 222 222 221 218 217 214 209 204 199 195 189 184 178 1 44 164 206 214 215 214 206 197 143 95 92 89 88 87 82 77 74 77 74 71 70 66 67 73 78 80 73 70 73 68 67 68 69 79 90 86 75 65 73 82 85 88 84 88 104 116 126 140 142 136 131 128 128 134 144 152 164 161 159 152 166 172 167 162 148 142 156 181 201 208 221 129 96 196 225 221 226 225 225 224 223 222 221 221 220 217 215 213 208 204 200 195 189 183 178 0 31 138 195 208 212 212 207 202 183 119 88 85 81 83 83 79 73 74 75 73 71 62 53 55 63 67 66 68 74 69 66 60 56 69 75 76 71 62 79 71 66 77 80 86 104 107 110 117 124 132 125 113 122 134 148 154 151 144 152 159 150 172 178 153 132 141 177 204 205 209 251 173 58 138 215 227 224 224 224 224 222 222 222 222 221 218 215 211 207 205 199 193 188 183 177 1 18 112 183 201 205 206 206 199 195 162 102 88 82 79 82 82 74 71 72 69 71 72 65 55 46 50 55 59 65 66 63 64 68 59 62 76 73 65 69 55 60 72 78 73 92 93 107 110 117 118 108 104 118 142 143 143 145 140 130 142 146 163 167 136 142 175 202 205 205 219 255 196 53 83 178 224 226 223 224 223 222 221 221 220 219 217 214 211 207 205 199 193 188 183 177
## 1862                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         173 172 172 172 171 172 173 173 173 174 174 174 174 174 174 174 174 174 174 174 172 175 162 88 44 45 29 29 42 31 10 15 27 23 22 30 42 61 89 112 132 135 141 144 148 151 151 151 151 149 134 115 96 81 59 52 53 42 33 23 17 17 11 8 4 13 25 28 21 7 3 4 3 3 6 7 16 22 6 1 2 1 0 0 1 3 3 3 3 3 3 2 7 9 6 2 173 172 172 172 171 172 173 173 173 174 174 174 174 174 174 174 174 174 174 173 171 163 153 97 54 49 36 23 28 31 26 27 27 33 50 65 78 95 126 145 149 152 151 151 153 157 158 159 159 159 144 125 105 88 64 47 46 36 32 26 20 15 7 12 17 12 20 19 19 15 9 11 9 8 7 7 18 25 17 11 11 7 1 0 0 1 0 0 0 0 0 1 5 6 4 1 173 172 172 172 171 172 173 173 173 174 174 174 174 174 174 174 174 174 174 175 177 169 162 119 67 52 47 40 34 41 47 50 46 52 75 103 127 138 150 153 150 154 154 154 156 157 158 159 160 158 148 136 121 101 74 55 48 34 28 19 15 14 14 22 19 22 26 9 10 17 11 13 18 21 23 20 19 21 17 11 10 9 4 2 1 0 1 1 1 1 0 1 3 3 3 1 173 172 172 172 171 172 173 173 173 174 174 174 174 174 175 175 175 175 175 172 177 179 172 132 66 55 58 50 51 62 66 70 81 96 116 141 156 156 156 156 160 162 160 159 160 161 162 161 157 154 143 139 130 112 83 61 53 39 34 25 17 15 12 14 9 9 14 10 10 9 7 10 14 17 18 12 5 4 2 3 3 3 3 4 2 0 0 0 0 0 1 3 1 0 1 3 173 172 172 172 171 172 173 173 173 174 174 174 174 174 175 175 175 175 175 174 170 177 172 137 76 63 58 56 64 78 97 112 125 142 153 162 165 160 162 160 161 163 162 162 163 162 161 156 151 149 143 138 130 114 88 63 56 47 45 35 34 34 29 26 24 13 12 19 17 11 12 11 8 4 2 1 3 3 2 4 3 1 1 0 0 2 2 2 2 2 3 4 1 0 1 4 172 172 172 172 173 173 174 174 174 173 173 173 173 173 173 173 173 173 173 175 174 182 177 160 98 72 77 101 104 114 137 150 154 164 163 163 162 161 163 161 160 163 161 160 161 163 160 154 150 149 144 137 130 116 90 68 54 50 56 50 48 47 36 32 30 20 16 19 13 9 13 6 12 11 8 4 4 3 3 3 2 0 3 2 2 4 2 2 3 4 3 1 0 0 0 1 172 172 172 172 173 173 174 174 175 173 173 173 173 173 175 175 175 175 175 175 174 176 176 157 91 79 117 151 156 157 171 176 172 172 166 164 162 161 161 160 160 162 161 161 163 166 158 149 148 151 150 148 132 118 96 72 57 45 55 49 47 46 40 42 42 33 23 14 5 5 8 14 30 19 13 9 6 5 4 3 2 3 4 2 0 1 1 1 1 2 2 0 0 0 0 0 172 172 172 172 173 174 174 175 175 175 175 175 175 175 177 177 177 177 177 178 178 179 179 145 88 106 161 179 181 184 187 183 173 170 168 166 163 161 161 161 161 161 161 162 164 165 158 150 148 152 153 152 126 118 113 86 69 47 45 54 53 48 50 53 48 33 23 16 10 10 8 10 16 7 8 8 6 5 3 3 3 4 3 2 0 0 0 0 1 1 1 1 1 1 1 1 173 173 173 173 173 173 174 174 175 176 176 176 176 176 176 176 176 176 176 176 177 178 175 138 114 154 189 187 188 192 187 178 171 172 170 169 165 163 162 162 162 162 163 163 161 158 157 151 148 148 151 155 134 117 109 90 83 70 47 51 50 44 49 52 52 41 27 11 8 12 12 9 8 9 13 6 1 1 1 3 4 1 0 2 3 2 1 1 0 0 0 1 1 1 1 1 173 173 173 173 173 174 174 175 175 176 176 176 176 176 175 175 175 175 175 177 173 172 173 156 161 196 199 197 196 194 189 180 177 173 171 170 167 165 165 166 163 163 165 163 157 153 154 150 147 148 151 153 145 120 106 102 85 69 56 51 48 39 37 40 52 51 40 24 16 11 2 4 7 5 9 2 1 0 1 1 2 1 0 2 3 1 2 1 1 0 0 1 1 1 1 1 174 174 175 174 174 175 175 176 176 176 176 176 176 176 176 176 176 176 176 179 176 181 172 173 193 200 204 202 195 192 187 183 180 175 172 171 168 167 166 165 164 164 163 160 156 152 151 150 149 150 149 148 139 129 109 103 95 77 72 54 56 50 37 30 42 46 44 39 29 19 21 28 30 15 17 30 16 9 8 5 11 11 5 1 2 3 0 2 0 0 0 1 2 2 1 0 174 174 175 176 175 175 176 176 176 177 177 177 177 177 177 177 177 177 177 177 175 177 173 190 204 201 205 201 193 190 185 182 179 174 170 168 166 164 164 164 166 165 163 161 158 152 150 149 150 152 148 138 132 136 114 97 112 106 90 68 71 66 56 42 36 42 48 45 31 19 18 27 40 41 42 47 31 26 31 22 20 26 16 4 6 3 1 2 2 1 1 1 0 0 1 3 174 175 176 176 175 175 176 176 177 178 178 178 178 178 177 177 177 177 177 175 180 176 181 197 205 206 202 197 193 190 186 185 183 176 171 168 166 164 164 164 165 163 161 159 158 156 152 149 150 152 147 133 124 128 120 106 102 110 110 91 70 64 68 55 38 36 44 57 55 37 13 3 10 35 47 33 22 24 27 14 9 18 10 3 2 1 3 0 3 4 2 1 3 4 2 2 175 176 175 176 176 176 177 177 177 178 178 178 178 178 177 177 177 177 177 178 180 174 188 204 204 203 198 194 193 187 181 181 179 173 170 171 168 167 166 165 161 159 157 155 156 160 155 152 149 149 146 137 122 115 119 114 99 105 109 93 77 67 58 52 52 42 31 39 51 42 25 20 19 32 49 51 54 47 31 16 5 2 12 20 6 4 3 1 1 2 3 7 3 1 2 5 176 175 175 176 176 176 177 177 178 177 177 177 177 177 177 177 177 177 177 177 175 176 197 205 200 202 195 192 188 183 180 176 175 172 171 172 169 167 167 166 161 159 156 155 156 157 157 156 153 152 148 139 128 117 112 109 114 113 105 90 77 79 56 48 50 50 43 32 32 38 31 20 14 23 43 56 63 54 41 30 9 1 13 22 14 9 8 7 3 4 7 7 6 3 2 2 175 176 177 177 177 177 176 176 176 177 177 177 177 177 176 176 177 177 178 179 173 184 201 200 199 199 192 187 185 181 179 176 176 174 171 170 167 166 164 163 161 159 157 157 156 155 158 159 156 154 152 144 130 118 116 117 116 110 110 97 74 72 62 43 44 52 53 52 41 28 29 26 17 21 32 41 48 46 35 31 18 14 5 7 12 7 2 0 0 0 0 0 4 8 4 2 176 178 177 177 177 177 176 176 175 177 177 177 177 176 176 177 177 177 178 179 176 191 203 198 196 194 188 183 182 181 177 174 173 171 169 169 166 164 163 161 157 157 157 159 159 158 161 159 155 151 155 151 135 119 115 130 123 104 99 93 80 62 58 35 37 50 47 58 69 53 28 21 22 22 28 41 53 52 39 32 32 29 15 19 14 5 15 32 55 60 45 26 5 0 3 7 177 177 178 178 177 177 177 177 176 179 178 178 177 176 177 177 177 178 178 177 181 197 203 195 192 191 185 181 179 179 175 172 171 168 166 166 164 162 161 159 156 156 156 158 159 159 160 159 155 152 150 146 139 128 120 113 114 104 95 82 80 67 48 40 39 42 46 42 53 64 53 34 26 24 30 44 51 59 62 41 36 30 20 24 26 59 109 130 138 133 125 117 76 26 0 1 178 178 178 177 177 177 178 178 177 179 179 179 179 179 177 178 178 179 179 179 185 199 198 194 192 189 186 181 180 178 172 169 169 167 166 164 163 161 161 160 159 157 156 155 156 156 157 157 155 154 147 137 133 132 128 113 100 93 98 92 76 71 59 44 54 56 60 57 47 51 51 42 45 39 27 32 34 45 68 55 42 32 4 27 98 141 149 130 117 113 109 123 142 118 54 9 178 178 177 178 177 177 178 178 179 180 180 181 182 183 179 179 179 180 180 178 188 201 194 190 190 186 185 181 177 174 169 166 167 167 166 164 162 161 162 162 161 159 157 156 156 156 156 154 153 152 147 135 136 135 123 119 105 93 91 90 77 58 68 58 53 64 67 58 58 62 55 49 43 33 30 34 35 42 48 51 36 12 18 101 151 122 76 55 42 43 50 64 114 148 125 71 179 179 177 178 178 177 179 180 180 179 182 184 184 184 180 180 180 180 180 178 190 202 194 189 188 185 184 178 176 172 170 169 169 168 166 164 163 162 162 162 159 160 158 158 158 158 158 154 150 148 144 139 143 141 129 121 120 116 102 92 76 49 57 73 66 59 68 58 63 70 66 62 38 29 29 39 46 46 41 36 17 12 74 153 111 46 26 50 56 69 95 84 95 125 138 118 179 179 178 178 178 178 180 180 179 180 183 186 185 183 181 182 182 182 182 181 191 201 194 190 188 185 184 177 176 171 170 168 169 166 163 163 162 162 162 161 159 159 159 157 157 159 160 155 149 146 148 145 145 147 149 139 132 130 127 117 99 76 61 58 75 61 63 57 52 60 68 67 42 37 33 43 46 36 37 28 13 53 131 133 53 32 74 112 117 116 126 120 115 122 136 134 180 179 179 178 177 180 180 179 178 182 186 188 187 184 181 182 183 184 185 181 191 200 195 191 189 185 184 177 176 171 168 166 166 161 160 162 162 160 159 158 157 157 157 156 154 153 157 154 150 151 149 144 144 149 154 148 140 140 141 134 120 104 83 62 69 63 65 59 45 56 62 64 49 47 44 50 46 30 30 35 60 98 137 104 27 44 114 144 134 126 123 121 123 121 135 142 180 180 179 179 178 180 181 180 180 186 188 190 189 186 183 183 183 184 184 182 191 200 195 191 189 185 184 177 176 171 170 168 168 162 163 165 163 160 155 153 153 153 153 152 151 151 155 151 146 147 147 144 144 147 148 145 144 139 136 137 126 111 98 79 70 71 67 60 54 54 48 54 49 48 49 53 49 34 29 62 115 112 111 87 22 41 111 144 152 151 126 117 119 125 141 145 181 180 180 180 179 180 180 183 186 186 189 192 191 188 188 188 188 187 187 184 192 201 195 192 189 185 184 176 176 171 170 169 169 166 163 165 165 160 155 152 149 149 148 149 148 150 155 147 139 143 147 144 146 148 147 142 142 140 137 138 127 109 102 88 73 71 66 62 65 56 51 55 46 33 42 48 43 50 49 91 124 105 114 94 17 33 98 133 159 170 143 119 108 112 127 142 181 180 180 181 181 180 181 185 188 188 191 194 193 192 194 193 192 191 189 185 193 201 195 195 191 184 183 180 178 172 170 169 168 166 164 163 163 164 162 155 146 147 146 146 146 146 148 145 142 145 145 142 142 145 146 140 136 138 140 135 124 114 110 94 75 64 63 63 69 57 59 65 41 21 34 37 36 66 85 111 120 105 117 96 27 53 112 122 137 162 157 124 114 113 113 140 181 179 181 183 182 183 183 185 186 191 194 196 198 198 197 195 194 193 191 189 196 200 195 197 193 186 183 184 180 174 170 168 165 161 165 168 165 168 165 156 145 141 140 141 144 144 141 141 143 144 145 145 145 143 140 140 137 139 141 131 124 116 111 103 82 64 62 70 75 56 57 65 38 25 29 27 32 69 104 115 117 110 119 100 51 72 121 122 132 148 148 124 112 110 106 135 182 180 181 184 186 186 186 187 188 188 194 200 205 205 201 196 196 198 194 196 200 199 196 196 191 186 183 178 177 175 171 166 165 164 166 167 161 159 151 144 139 129 128 131 134 137 134 133 138 143 142 146 146 142 137 138 137 139 139 130 125 117 112 104 83 67 64 76 72 48 57 60 42 36 35 36 49 81 107 116 112 106 121 99 73 106 117 119 144 154 151 132 109 97 101 131 184 182 183 186 189 190 191 192 191 185 192 205 212 212 204 200 199 200 199 202 201 198 196 194 187 186 184 181 180 175 167 162 163 165 165 163 154 145 134 130 134 130 130 128 126 129 127 126 132 142 143 144 144 141 138 137 133 132 132 128 124 118 110 96 74 69 63 69 63 42 59 57 41 42 37 54 87 107 109 117 119 116 125 93 83 146 153 158 167 156 154 159 125 94 91 118 187 185 186 189 190 192 194 196 195 192 198 212 218 212 203 202 199 197 199 202 198 196 196 190 185 185 185 183 176 168 159 153 153 153 149 145 135 127 121 121 129 130 126 123 122 121 123 121 127 137 138 139 138 138 137 135 132 129 128 124 124 123 113 92 73 70 60 61 64 52 61 56 54 64 64 82 114 123 118 129 125 122 131 102 51 66 111 133 136 137 149 171 146 101 82 103 187 186 187 190 190 192 195 199 199 201 207 218 219 213 204 203 199 193 198 200 194 195 194 185 177 174 174 170 159 146 136 129 127 121 113 111 109 110 115 118 118 117 110 112 121 124 122 119 125 133 131 132 134 134 133 134 133 132 129 125 124 126 117 97 83 89 83 83 93 88 92 90 91 96 93 104 125 129 127 130 118 118 123 114 55 9 53 63 66 108 135 153 150 111 82 102 187 185 186 189 190 190 192 198 202 205 213 219 217 217 209 205 198 193 193 197 191 195 194 180 169 160 153 145 131 117 107 100 96 87 78 79 87 93 98 102 103 105 101 107 120 124 115 114 123 129 126 125 129 131 131 134 134 133 130 128 125 127 122 112 109 119 117 112 115 115 114 114 108 104 104 112 127 135 131 116 100 100 108 115 82 35 102 118 94 107 109 125 148 128 86 98 192 190 190 191 192 193 190 193 201 211 215 216 217 219 213 206 199 192 192 195 191 195 190 173 158 145 133 122 104 92 84 76 70 64 67 75 79 79 79 80 85 92 90 87 93 105 107 115 123 124 120 120 126 130 130 135 134 133 131 132 130 132 129 125 126 128 123 116 112 111 114 113 108 105 110 120 130 131 125 112 100 102 95 106 109 65 89 124 133 138 110 91 128 140 91 93 200 200 201 201 199 200 195 195 204 210 212 211 213 216 215 209 204 200 198 194 190 192 181 160 147 132 115 101 80 74 68 62 61 60 68 76 77 78 80 84 84 85 88 94 102 107 108 111 116 115 116 117 125 130 131 136 135 134 134 133 133 132 127 124 127 122 120 115 108 109 113 113 106 105 113 121 127 125 122 120 116 116 90 90 115 89 83 101 108 128 130 102 120 140 96 95 204 206 208 207 206 207 203 202 205 206 209 209 210 214 214 211 209 207 202 198 189 184 164 139 126 113 94 75 58 56 54 56 61 65 76 79 81 80 82 86 87 88 91 91 90 94 92 98 107 111 117 119 127 132 134 137 138 138 136 132 132 131 130 128 128 125 124 119 111 108 112 112 108 108 114 119 123 126 128 130 132 134 100 81 98 104 102 111 114 125 150 151 146 134 95 99 212 213 213 212 209 208 208 205 202 205 208 210 211 214 214 211 211 209 199 194 184 168 141 112 96 84 70 55 45 41 36 46 56 67 78 76 81 82 82 87 89 83 80 77 72 70 80 94 108 116 119 122 130 135 136 136 140 140 136 131 132 131 129 127 123 121 120 116 112 109 108 109 106 105 106 115 123 122 130 143 147 147 138 108 85 92 98 111 125 131 152 160 162 125 85 104 218 213 211 212 210 207 204 202 202 203 206 210 214 215 210 210 215 215 194 192 186 164 128 94 79 66 57 46 32 26 30 60 78 83 73 51 46 45 49 53 64 72 69 66 72 84 92 101 110 120 131 133 140 144 141 135 136 136 133 135 135 132 129 125 122 120 119 116 110 109 110 110 104 101 102 110 123 125 134 146 153 156 164 140 74 65 97 103 111 116 124 139 150 109 79 112 222 217 216 214 211 208 204 201 201 202 207 212 213 212 209 214 214 218 185 174 190 164 122 94 81 66 55 44 31 28 35 61 65 46 28 10 14 22 15 14 31 49 56 72 96 103 97 106 129 145 149 144 143 144 144 142 137 131 131 137 136 135 133 129 125 120 117 115 112 110 108 106 100 97 101 111 121 127 141 148 158 160 170 149 78 46 81 98 103 117 128 143 133 96 90 118 229 225 222 219 214 210 205 202 201 204 209 214 213 211 212 209 210 222 170 139 189 173 128 103 88 73 62 56 48 42 45 48 29 6 9 47 84 109 94 77 79 84 85 89 96 104 119 129 139 148 148 146 142 140 138 143 139 132 130 135 135 135 134 129 124 114 110 110 112 110 107 106 102 97 103 113 119 128 144 147 156 162 164 132 79 38 66 115 127 140 148 145 114 86 111 136 232 230 227 225 218 213 209 207 209 209 213 215 215 214 215 212 216 222 158 100 153 186 154 110 93 76 62 58 50 48 58 57 33 5 30 103 111 136 122 116 122 121 129 125 116 127 143 143 138 143 146 149 148 144 143 139 139 135 130 130 130 130 127 122 117 110 104 107 111 110 108 110 105 99 105 110 112 125 140 141 154 169 163 99 43 37 96 157 158 148 145 124 101 100 130 134 234 232 230 228 222 217 213 214 215 214 215 217 217 218 210 210 214 219 178 81 94 179 174 130 106 78 67 65 60 60 68 68 57 25 47 102 73 117 129 136 127 121 130 138 136 136 143 143 143 147 147 150 148 145 145 137 138 135 130 129 127 123 118 114 109 108 104 106 113 112 109 111 109 100 102 108 114 130 145 152 166 177 167 80 20 68 146 175 150 126 133 124 107 121 139 122 236 233 231 230 227 222 220 221 219 217 220 221 220 219 216 216 218 221 208 101 45 124 178 161 133 103 82 73 76 81 82 74 64 59 56 74 108 147 153 136 126 132 147 153 145 135 139 142 144 145 147 150 150 152 147 138 133 129 128 126 125 119 112 109 106 101 98 101 109 110 110 118 116 103 102 105 114 131 152 165 165 163 151 104 89 142 174 163 134 114 131 130 109 124 134 111 238 237 237 236 231 229 228 228 225 224 226 226 224 222 220 217 219 213 217 176 81 48 150 174 143 136 116 101 100 101 104 91 85 85 63 93 148 155 144 126 124 137 146 149 149 148 146 147 147 149 153 155 155 155 147 137 132 126 125 121 119 113 109 107 106 100 94 96 102 103 108 120 120 108 105 106 109 127 144 145 143 135 125 133 159 167 138 132 122 111 141 121 111 126 128 98 239 237 237 236 233 232 232 232 230 228 230 230 227 224 219 216 215 212 210 215 196 81 69 163 169 154 147 135 127 130 133 118 127 110 85 141 158 141 112 108 129 151 156 149 147 146 147 147 148 149 152 154 155 155 145 132 127 121 119 115 108 106 107 107 101 105 100 99 101 103 110 120 124 113 104 103 103 122 136 129 126 121 103 115 143 132 105 111 120 114 130 111 118 135 117 75 239 236 235 234 235 233 234 233 229 228 229 230 228 225 223 221 216 221 207 205 216 189 87 147 193 166 158 147 139 153 152 139 145 134 117 159 153 105 71 101 140 159 164 158 153 144 148 151 153 153 157 158 159 159 146 130 124 118 115 112 104 105 110 110 102 110 105 103 104 108 125 131 124 112 110 105 101 115 129 125 115 110 103 107 127 131 120 117 127 125 121 124 138 136 92 44 239 238 238 238 237 236 236 233 230 227 229 230 230 228 224 218 214 217 209 203 198 211 183 173 182 180 160 142 152 169 150 137 142 132 128 118 99 90 118 154 163 163 160 159 162 159 159 160 162 163 162 160 159 156 139 130 126 120 118 114 112 112 114 115 113 108 98 100 104 114 128 131 126 112 103 100 102 108 120 123 111 101 104 123 142 156 143 132 150 146 135 143 134 100 56 34 239 238 238 238 237 236 235 233 229 230 230 230 228 227 224 219 215 211 209 205 199 197 198 187 184 179 161 151 164 170 150 138 138 130 125 119 125 144 167 172 165 166 165 163 161 157 159 159 160 165 163 159 158 151 137 128 124 119 116 114 113 112 113 111 110 108 104 107 107 114 124 126 123 110 98 97 101 102 118 121 117 101 110 139 149 159 144 132 149 146 138 141 111 70 44 36 239 238 238 238 237 236 235 234 230 231 230 228 227 224 221 217 213 209 208 203 197 201 198 189 188 177 154 149 161 163 161 155 150 146 145 159 171 175 178 177 172 173 170 167 163 161 163 161 158 158 157 151 153 149 134 127 123 118 114 112 112 112 109 107 102 109 112 112 110 112 120 121 114 105 99 98 103 105 109 115 120 101 110 155 163 159 150 140 145 135 126 115 77 45 33 36 239 239 238 238 238 237 235 234 230 230 229 225 222 222 217 214 210 207 206 203 197 200 197 187 183 175 153 154 162 157 169 167 158 162 161 167 176 181 186 183 176 177 173 170 167 163 164 161 153 148 145 138 145 147 132 123 120 115 110 108 111 112 109 106 103 110 114 115 116 114 117 118 114 109 104 104 108 106 104 112 111 97 106 146 166 155 150 145 136 112 98 84 44 19 26 38 240 239 239 239 238 237 237 235 231 230 227 222 219 220 211 209 207 205 205 204 199 200 191 180 178 173 158 160 164 159 172 169 161 172 167 165 176 186 180 170 169 171 171 169 166 162 159 154 148 144 139 133 140 141 129 121 118 115 112 111 113 113 111 111 111 111 114 116 120 119 117 119 120 118 111 105 102 99 104 116 103 100 106 123 154 156 140 130 115 96 79 54 30 19 28 37 240 240 239 239 239 237 236 235 231 230 226 221 218 218 207 205 204 205 206 205 204 203 189 178 182 173 161 158 160 166 175 174 172 177 176 179 185 184 181 172 161 165 165 162 160 158 150 146 145 144 141 136 137 134 123 121 119 119 119 120 116 114 112 112 112 112 115 115 120 120 115 116 119 119 111 102 99 97 96 108 99 99 95 93 109 130 123 106 92 69 42 25 26 28 31 34 239 239 238 238 238 238 238 236 233 230 225 221 217 213 206 202 206 210 208 206 207 201 188 177 179 179 168 164 170 172 179 176 171 179 183 185 186 183 178 171 161 162 162 158 154 153 147 142 142 138 137 133 133 129 122 118 121 124 123 121 114 112 111 113 116 115 116 116 118 122 117 113 114 113 107 102 99 97 95 95 95 93 84 79 78 79 71 60 49 30 13 24 28 31 28 30 239 238 238 237 237 237 238 235 234 230 225 223 217 210 208 205 208 213 211 209 210 199 182 172 173 179 179 179 176 170 172 174 179 184 182 183 183 180 174 168 162 161 161 156 150 147 142 137 138 135 132 128 126 122 118 115 119 123 120 118 116 115 115 116 116 112 119 120 114 117 117 113 113 110 108 104 99 97 95 93 89 87 88 85 75 61 41 28 22 23 23 28 31 30 26 34 238 237 237 237 236 236 236 235 233 230 227 225 219 211 212 210 212 215 214 215 212 194 176 170 167 178 186 179 170 165 165 171 181 183 181 183 182 177 171 165 159 157 155 151 145 138 133 130 134 132 128 124 119 116 115 116 120 122 121 120 120 121 121 118 115 113 117 118 114 113 115 114 110 108 110 106 98 95 93 90 86 87 84 79 70 63 56 42 33 29 26 27 33 31 32 48 237 236 236 235 235 236 235 234 231 229 227 226 221 214 215 215 217 217 217 220 209 182 173 171 163 175 182 164 155 161 164 169 175 182 185 181 178 172 167 162 155 150 147 142 136 130 127 127 133 131 125 120 115 113 117 123 125 125 126 126 123 123 122 120 117 119 114 113 119 115 112 112 110 109 108 105 96 92 93 88 84 83 80 75 66 59 63 67 66 56 45 41 45 48 56 75 236 236 235 235 235 235 235 232 231 229 226 225 222 218 217 218 219 218 218 219 200 169 169 171 163 167 172 158 145 146 143 154 172 178 178 175 171 165 161 157 151 145 141 134 127 126 128 130 135 131 124 118 113 114 122 128 128 128 128 127 126 125 124 125 125 123 119 119 120 114 106 108 113 110 105 101 93 89 92 89 81 78 77 74 64 59 68 74 79 81 81 82 75 81 90 100 234 235 235 234 234 233 233 232 231 229 227 225 222 222 217 218 216 217 220 214 188 157 164 169 164 163 159 159 145 131 122 133 160 174 170 171 168 160 153 151 147 139 132 128 124 123 128 128 131 128 124 124 117 116 126 130 130 129 127 127 128 128 128 126 130 123 121 124 117 108 108 112 110 111 102 99 96 91 93 88 80 81 76 71 65 65 71 73 76 86 101 105 98 102 109 110 233 232 233 233 232 232 231 231 231 228 227 225 222 221 217 218 215 217 220 207 178 157 162 168 166 158 153 157 151 136 121 117 130 163 172 167 165 159 149 142 141 133 126 124 121 117 119 121 124 121 124 127 123 124 133 136 136 136 133 128 135 138 132 130 128 126 122 116 115 115 112 110 107 109 101 100 100 95 92 87 80 80 76 72 70 69 71 76 79 85 95 100 98 101 106 107 231 231 230 230 229 229 229 228 228 227 226 223 221 219 218 219 216 216 218 199 171 161 158 164 160 146 147 150 149 152 151 139 124 140 163 160 151 147 142 134 134 129 123 120 118 114 114 120 127 125 124 124 126 132 137 137 135 137 136 127 129 132 129 134 127 131 122 112 120 111 104 106 108 107 100 100 102 97 89 86 82 78 76 73 72 70 72 80 82 84 89 91 96 100 104 106 230 230 228 228 228 227 227 226 226 226 224 222 220 219 216 217 214 213 214 192 170 164 152 156 148 135 141 148 149 143 144 144 126 122 148 159 145 136 138 133 130 127 123 118 116 112 114 125 130 129 125 125 126 130 130 131 128 130 131 127 119 122 132 141 135 127 125 114 111 110 104 103 112 108 101 100 102 97 87 84 83 79 76 72 70 69 71 78 81 85 91 90 96 102 106 108 230 229 227 227 227 226 226 225 225 225 224 221 219 219 212 213 211 207 208 185 171 166 150 153 146 136 136 151 156 144 137 133 120 102 122 148 147 134 137 134 126 124 120 115 114 114 120 130 128 128 127 128 127 127 129 129 130 131 129 123 123 126 143 143 135 128 130 115 107 114 109 101 111 103 99 100 99 93 86 85 82 77 74 70 68 68 71 79 81 84 91 94 94 102 108 108 229 228 226 225 224 224 225 224 223 222 222 219 218 216 212 213 208 211 207 178 169 162 154 156 145 137 144 162 165 156 144 132 120 88 97 131 146 137 130 130 124 123 119 115 115 122 129 132 127 127 127 126 125 126 129 130 134 134 130 124 130 133 143 134 127 133 127 118 114 107 103 98 101 95 95 96 96 92 86 85 79 75 72 72 70 67 73 79 81 83 88 93 94 102 109 107 228 227 225 224 222 222 223 221 221 220 219 218 216 213 212 212 205 211 205 173 162 161 168 165 147 145 158 168 165 155 136 121 109 84 97 127 138 136 129 128 130 125 122 123 126 129 134 128 123 126 124 124 123 124 123 126 131 129 128 131 133 135 134 129 130 133 121 116 110 105 102 92 92 89 94 92 90 85 79 79 77 74 71 74 71 66 74 81 81 84 90 95 94 100 108 108 227 226 223 223 224 224 223 222 221 220 218 215 212 210 209 209 203 209 200 171 153 150 159 165 150 148 154 155 142 127 108 87 70 64 111 147 135 133 138 133 131 127 126 130 135 137 139 128 120 124 121 121 122 123 121 122 125 123 121 123 122 133 134 134 132 126 122 115 109 108 106 96 94 89 92 88 82 77 74 76 76 71 70 71 70 67 73 82 81 84 89 95 93 98 107 108 227 225 223 222 219 218 215 215 215 215 214 212 209 208 207 209 208 214 198 169 153 143 137 139 141 132 124 118 99 73 44 33 50 82 132 158 139 132 142 140 136 135 136 138 138 139 141 132 121 121 121 117 119 123 125 124 120 123 127 117 113 128 132 131 129 123 124 121 117 113 102 90 94 91 91 84 78 80 80 76 73 70 69 69 69 67 71 79 81 83 89 90 94 105 112 109 225 224 222 220 213 212 208 208 208 211 211 210 208 206 206 207 205 214 202 169 151 141 125 115 107 97 89 57 23 13 13 39 84 113 119 138 148 138 136 142 142 142 143 141 141 139 138 131 125 124 124 119 121 125 124 125 124 125 125 119 111 121 130 127 122 123 125 121 121 117 101 86 89 90 92 84 79 85 84 76 74 72 70 67 66 67 70 79 83 88 93 92 98 110 115 111 225 223 220 218 214 213 210 209 208 208 208 207 206 205 207 203 201 210 204 169 141 132 118 98 81 66 49 15 9 28 74 116 114 97 87 111 145 149 144 147 148 145 143 141 141 143 133 129 129 129 126 122 124 124 120 122 124 117 111 114 112 121 131 123 119 128 125 119 120 113 102 91 89 88 87 83 81 85 83 77 75 73 69 66 64 66 72 81 85 91 97 98 103 111 115 110 222 220 217 214 212 210 208 206 206 207 206 205 204 203 204 204 204 209 206 176 132 115 107 82 73 54 23 10 36 70 106 112 83 67 75 85 107 136 150 154 159 153 151 147 143 143 130 126 125 127 129 125 123 121 119 123 127 123 115 110 116 121 115 111 118 127 118 113 114 101 92 89 87 88 85 80 81 84 83 78 74 70 67 64 64 64 71 79 83 87 93 98 103 109 111 107 220 218 214 212 209 206 204 204 206 205 205 205 203 202 205 203 199 204 209 196 145 94 69 63 58 35 26 55 68 79 85 71 75 71 75 78 85 106 135 153 149 150 150 147 141 134 127 124 124 126 130 129 125 123 124 126 126 124 121 117 115 114 113 114 117 116 108 104 104 96 91 88 87 90 85 79 80 81 81 78 75 70 65 62 63 65 70 76 79 81 87 95 104 107 104 104 221 218 215 212 207 205 202 203 204 202 203 203 201 199 201 201 195 199 206 205 193 154 96 61 51 72 115 167 130 73 78 79 87 84 80 74 81 92 114 136 137 142 147 144 139 134 129 125 126 129 131 133 129 125 126 124 117 112 115 120 120 114 113 116 121 112 104 102 102 99 94 86 86 89 84 79 79 78 78 77 77 75 64 58 61 65 69 75 75 78 82 88 102 108 108 110 219 216 214 211 209 208 204 203 203 203 204 205 202 199 198 202 201 203 201 195 206 210 196 179 179 195 193 164 106 74 102 99 78 78 77 66 74 85 93 112 127 133 143 144 140 136 133 126 129 135 136 135 132 125 118 118 122 119 112 107 114 113 109 108 109 110 108 103 98 96 96 88 83 88 84 78 78 76 76 76 76 68 61 56 58 62 68 75 76 79 84 90 105 114 115 116 216 214 211 209 209 208 206 205 203 206 207 208 206 202 203 204 202 204 203 197 194 193 197 199 192 159 118 90 87 93 108 100 78 71 60 55 60 60 63 79 91 104 119 127 133 130 131 129 135 142 134 135 138 130 121 119 124 122 111 108 104 108 112 110 104 111 111 103 96 90 95 90 82 84 82 77 78 76 74 74 69 60 58 56 57 60 67 75 77 81 88 102 111 115 113 111 214 213 211 210 208 206 206 206 202 204 205 205 204 202 203 204 204 203 204 203 197 194 191 179 156 116 90 91 87 89 96 89 83 78 62 57 55 52 55 58 68 88 99 111 135 135 135 134 137 137 134 138 143 136 130 129 123 116 108 116 113 105 105 111 107 109 108 102 99 90 88 86 80 77 78 76 78 75 71 67 61 56 54 53 57 62 70 75 77 81 89 103 107 108 109 111 212 212 210 209 207 206 205 205 202 203 203 203 202 201 203 204 204 206 206 209 205 200 186 173 155 124 106 103 108 108 95 76 77 80 67 63 62 57 55 51 51 69 73 85 113 129 132 133 137 138 133 131 135 132 126 125 123 118 111 115 117 105 107 112 101 98 100 96 91 84 82 80 75 76 76 75 75 70 67 62 52 49 52 55 60 65 74 76 76 81 93 106 106 105 108 112 210 211 210 209 209 207 206 206 202 202 201 201 200 200 204 203 203 206 208 208 205 200 184 175 170 131 110 118 134 126 108 87 78 77 70 66 66 64 59 55 55 59 63 65 79 110 127 136 140 136 129 129 132 131 122 115 115 113 106 109 112 100 105 110 101 95 94 94 91 88 82 77 77 78 77 72 66 62 63 55 47 49 56 61 63 67 76 77 78 85 97 105 107 108 109 111 210 211 211 210 211 208 208 208 203 201 200 198 199 199 204 203 203 204 207 206 203 201 184 177 175 137 120 122 128 115 105 93 82 78 71 70 74 74 67 60 57 60 63 52 51 87 117 134 136 127 125 127 126 125 117 110 104 99 97 108 111 100 105 107 107 101 98 96 91 91 88 82 78 76 76 69 60 59 56 47 43 48 59 65 66 67 76 79 81 90 100 104 110 112 114 115 210 211 211 212 212 210 209 209 205 202 200 198 199 200 204 202 201 202 205 205 203 205 188 177 168 142 134 118 97 82 77 74 74 74 73 70 71 74 63 55 58 68 61 38 31 60 95 120 127 119 118 116 110 110 108 103 100 97 95 103 105 108 118 112 107 107 103 95 89 89 88 82 75 73 73 68 58 57 50 40 40 46 58 67 68 68 76 80 84 93 102 107 111 114 116 119 210 209 209 211 211 211 210 208 207 203 200 198 199 201 199 200 200 201 201 200 200 206 199 190 187 158 136 119 59 39 53 43 44 59 70 57 51 64 62 59 66 70 62 42 37 47 70 109 125 116 114 109 100 96 99 103 103 104 106 109 111 114 115 112 109 104 95 86 83 83 80 77 73 70 67 63 52 47 43 38 46 54 62 68 69 70 78 82 87 94 103 112 112 113 114 116 209 208 208 208 209 211 211 211 209 203 200 198 199 200 199 199 199 200 200 200 199 201 200 199 200 185 166 166 128 73 72 43 22 19 24 56 70 83 83 75 80 64 59 53 51 47 50 87 111 114 114 106 94 91 94 99 103 107 112 114 115 115 113 109 104 97 92 88 85 82 79 75 73 69 62 54 45 42 42 42 53 60 63 66 67 69 78 82 87 96 103 106 109 113 117 120 208 206 205 205 208 209 211 212 211 204 201 198 198 199 198 198 198 197 197 197 196 196 197 201 202 200 197 196 197 151 117 74 21 11 48 105 119 106 95 84 87 73 64 62 62 65 52 59 94 112 106 101 92 90 95 99 103 111 120 121 118 117 112 106 101 98 95 92 90 85 78 72 68 62 56 50 45 43 47 52 60 64 64 65 66 69 76 81 88 97 104 108 110 114 115 116 206 204 204 204 206 208 211 211 211 204 201 198 198 199 198 197 197 197 196 195 197 195 198 200 195 199 210 201 205 191 160 110 63 87 141 153 121 92 87 90 91 83 78 78 72 75 63 49 80 101 101 97 94 93 96 98 99 112 124 128 120 116 112 106 104 99 95 90 84 76 74 70 63 57 52 47 45 46 51 58 61 59 60 62 64 68 74 80 87 96 105 108 110 114 115 114 205 203 203 203 206 207 209 209 209 205 202 198 198 198 197 197 196 196 195 195 199 198 202 203 201 198 204 205 200 195 190 138 102 147 161 124 86 82 86 100 107 88 87 91 74 65 60 56 72 89 101 102 103 106 111 111 108 115 125 127 120 116 112 107 108 103 94 89 80 72 69 66 62 58 49 42 43 45 49 53 52 52 53 56 59 64 74 79 86 94 103 107 110 114 115 114 203 201 200 201 205 206 207 207 206 203 202 200 197 195 196 197 197 198 198 198 199 201 203 203 202 201 200 200 200 193 199 171 142 165 133 86 79 85 95 115 125 99 90 94 70 61 60 63 67 81 94 97 101 107 115 118 113 115 122 123 120 120 115 107 105 100 94 86 79 71 65 60 56 52 44 40 40 41 48 49 44 45 50 49 51 56 66 72 78 90 97 108 112 115 111 112 202 200 199 200 202 205 207 205 203 201 200 198 197 195 196 197 197 198 198 198 197 199 200 201 199 198 198 198 200 194 195 190 184 153 94 71 73 79 106 131 135 110 105 96 69 66 61 63 63 69 84 87 95 102 106 108 107 110 117 121 124 125 117 106 100 96 89 82 76 69 63 56 46 38 36 34 33 33 36 38 37 35 41 41 47 52 59 66 73 85 93 99 115 113 114 115 202 199 199 199 200 203 206 203 199 198 198 196 196 194 196 197 197 198 198 197 197 198 198 199 198 197 197 199 201 203 195 189 189 114 49 47 51 74 116 135 132 116 107 97 79 67 55 54 57 57 72 78 93 103 104 106 108 113 118 121 127 124 113 104 102 92 82 76 69 61 56 50 38 29 27 25 23 24 22 24 27 33 40 46 55 64 70 76 81 91 95 106 112 110 112 111 202 200 199 199 200 203 204 201 199 198 198 196 196 195 196 197 197 198 198 197 196 198 197 197 199 199 199 200 203 202 195 192 187 127 81 68 57 80 111 130 131 122 101 88 87 75 59 53 56 55 63 64 74 90 100 111 115 116 113 113 119 115 102 94 98 91 81 73 65 58 50 39 29 24 19 16 18 22 25 30 37 47 56 62 72 81 89 92 93 101 103 110 102 106 104 124 203 201 200 200 202 201 202 200 200 197 197 197 196 196 196 197 197 198 198 198 197 197 198 197 198 198 196 197 200 198 197 196 190 179 164 132 98 87 89 114 105 91 102 98 83 79 65 53 50 46 56 62 69 80 90 104 111 112 109 108 109 104 92 84 87 83 74 65 57 48 42 29 17 18 16 12 18 29 41 50 56 64 73 77 83 89 95 97 97 103 107 110 106 113 108 166 205 205 204 203 200 202 202 199 197 195 195 195 195 195 195 195 195 195 195 197 197 197 197 197 197 198 196 197 202 203 193 198 190 186 178 162 130 85 82 100 81 75 94 106 87 81 73 62 48 40 46 58 67 76 85 100 106 109 107 104 99 96 91 85 75 74 70 60 52 42 35 27 18 15 19 19 24 40 54 61 63 65 73 78 82 88 90 94 96 102 107 111 115 107 130 197 206 205 206 203 201 201 200 198 197 195 194 194 195 194 194 194 194 194 194 196 196 196 196 196 196 197 196 197 201 202 193 197 195 200 187 155 113 81 84 96 89 97 100 101 98 87 78 69 55 44 46 44 57 76 85 93 98 103 102 98 97 96 93 84 71 63 62 56 43 30 24 24 22 16 16 19 29 44 52 58 61 62 69 75 77 85 89 90 93 99 109 112 110 105 165 203 206 207 205 203 202 201 199 198 196 195 195 194 194 194 194 194 194 194 194 195 195 195 195 195 195 195 195 196 199 200 197 200 198 204 184 144 99 85 83 88 102 110 100 87 90 86 75 74 67 54 49 39 47 64 77 86 87 95 99 100 100 96 90 80 71 60 58 52 36 27 23 23 20 12 13 18 27 36 37 40 46 50 60 68 72 80 87 90 90 93 99 104 95 123 201 191 206 204 204 202 201 199 197 195 194 195 193 194 194 193 194 194 194 194 194 194 194 194 194 194 195 194 194 198 199 199 200 200 196 191 160 135 116 90 81 85 93 99 91 75 81 72 62 65 68 56 42 36 38 44 61 77 84 93 96 100 92 82 72 67 64 59 53 41 27 21 23 24 19 13 16 17 16 17 17 19 25 29 41 50 51 56 66 75 79 83 92 90 84 164 212 168 204 203 203 202 201 198 195 194 194 193 193 193 193 193 194 194 194 194 194 194 194 194 194 194 194 192 193 198 198 199 198 195 196 195 144 106 111 92 88 79 80 89 80 64 82 76 63 63 68 60 45 39 42 39 49 70 87 96 90 88 82 70 61 58 60 50 40 29 24 16 16 22 25 24 18 12 6 3 6 12 19 25 36 39 29 31 49 59 63 69 84 76 103 202 193 165 202 201 201 200 198 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 194 194 195 196 197 196 195 198 176 118 97 112 93 88 76 79 81 74 67 85 74 62 60 63 65 60 48 50 43 34 51 75 87 84 80 79 71 63 58 54 40 36 27 17 27 36 28 30 30 15 13 15 8 4 18 36 44 46 40 32 34 45 43 44 53 63 60 148 223 173 171 200 200 199 199 197 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 195 194 196 196 199 198 198 200 164 111 113 111 97 84 81 82 80 77 79 94 74 62 53 49 49 47 40 41 53 40 36 52 61 75 74 69 71 64 50 42 35 27 23 26 35 40 27 11 3 5 12 17 22 16 29 55 62 59 50 52 60 56 50 48 46 35 78 202 207 165 182 198 197 197 197 197 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 195 194 195 196 196 194 197 199 180 139 120 101 106 103 88 81 86 88 81 89 86 68 51 50 49 49 46 43 54 46 41 42 43 56 68 65 63 56 45 41 32 21 32 45 27 10 13 15 20 14 5 2 21 32 34 45 52 51 45 53 66 68 67 59 52 43 129 222 183 169 198 198 197 197 197 197 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 195 194 195 196 194 195 198 197 197 181 141 108 114 120 97 88 82 83 73 73 75 69 60 58 54 49 49 49 42 43 45 40 40 36 51 58 55 51 41 29 24 34 36 24 40 78 116 152 162 145 114 73 43 17 19 35 45 43 41 45 52 61 61 63 55 70 185 210 170 172 207
## 6493                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            128 128 128 128 128 128 128 128 128 129 133 133 133 132 132 132 132 132 132 133 133 133 133 133 133 132 132 131 131 136 140 142 142 142 142 142 142 143 143 143 143 143 142 143 143 143 143 143 143 142 142 141 139 139 144 149 148 148 148 148 148 148 148 148 148 148 148 149 155 154 152 152 155 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 161 159 158 164 168 167 128 128 128 128 128 128 128 128 128 129 133 133 133 132 132 132 132 132 132 132 132 132 132 132 133 132 132 132 136 141 144 142 142 142 142 142 142 143 143 143 143 143 143 142 142 142 142 142 142 142 142 143 144 144 146 148 148 148 148 148 148 147 147 147 147 147 147 148 155 154 152 152 155 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 164 164 164 166 167 167 128 128 128 128 128 128 128 129 129 130 132 132 132 132 133 133 132 133 132 133 133 134 135 134 135 132 135 137 140 143 144 142 142 142 142 142 142 143 143 143 143 143 143 143 143 143 143 143 144 145 145 146 148 148 148 148 148 148 148 148 149 150 150 150 150 150 150 152 156 155 154 154 156 157 157 157 157 157 157 158 158 158 158 158 159 160 160 162 163 163 165 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 133 132 136 138 135 138 133 140 143 142 142 142 142 142 142 142 142 142 143 143 143 143 143 143 143 143 143 143 142 145 148 148 148 147 147 147 148 148 148 148 148 150 153 153 153 153 153 156 158 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 132 131 135 137 134 137 132 139 142 142 142 142 142 142 142 142 142 142 143 143 143 143 143 143 143 143 142 143 143 145 148 148 148 148 148 148 148 148 148 148 148 150 152 152 152 152 152 155 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 132 131 133 136 138 136 138 137 140 142 142 142 142 143 143 143 143 143 143 143 143 143 143 143 145 144 144 145 145 145 146 148 148 148 148 148 147 148 148 148 148 147 150 155 157 156 155 155 156 157 157 157 157 157 157 157 157 157 157 157 157 158 158 159 160 160 161 163 163 163 163 163 165 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 131 128 134 137 137 137 139 143 142 142 142 142 142 143 143 143 143 143 143 143 143 143 143 142 146 147 146 150 148 148 148 147 147 147 147 147 147 148 148 148 148 148 146 147 153 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 163 163 163 163 163 165 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 131 128 134 137 137 137 139 142 142 142 142 142 142 143 143 143 143 143 142 142 142 142 143 142 147 146 145 149 147 147 148 147 147 147 147 147 147 147 147 147 147 148 146 145 151 155 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 163 162 162 162 162 166 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 132 130 133 136 137 137 139 142 142 142 142 142 142 143 143 143 143 143 144 145 145 145 145 145 148 147 147 149 148 148 148 147 147 147 147 147 147 148 148 148 148 148 150 153 157 158 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 161 161 164 164 164 164 166 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 132 132 134 137 137 139 142 142 142 142 142 142 142 142 142 142 142 145 148 148 148 148 148 147 148 148 147 148 148 148 147 147 147 147 147 148 148 147 148 148 147 149 152 152 154 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 160 160 166 168 168 168 168 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 132 132 134 137 137 139 142 142 142 142 142 142 143 143 143 143 143 145 148 148 148 148 148 147 148 148 148 148 148 148 147 147 147 147 147 148 147 147 147 147 147 149 152 152 154 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 160 159 165 167 167 167 167 167 167 167 167 167 128 130 130 130 130 130 130 132 132 132 132 132 132 132 133 132 131 133 135 134 134 136 139 139 140 142 142 142 142 142 142 143 143 143 143 143 145 148 147 147 148 148 147 148 148 148 148 148 148 147 147 147 147 147 147 148 148 148 148 151 153 154 154 155 157 157 157 157 157 157 157 157 157 158 158 158 158 158 159 160 160 162 163 163 162 162 166 167 167 167 168 168 168 168 168 168 129 132 132 132 132 132 132 132 132 132 132 132 132 133 133 131 129 135 137 137 137 139 142 142 142 143 143 143 143 143 142 143 143 143 143 143 144 148 148 148 148 148 148 147 147 147 147 147 147 148 148 148 148 148 148 147 147 147 149 156 158 157 157 157 157 157 158 158 158 158 158 158 157 158 158 158 158 158 161 163 163 163 163 163 165 168 167 167 167 167 167 167 167 167 167 167 129 132 132 132 132 132 132 132 132 132 132 132 132 132 133 130 128 134 138 136 137 139 142 142 142 143 143 143 143 143 143 142 142 142 142 142 144 148 148 148 148 148 148 147 147 147 147 147 147 148 148 148 148 148 148 147 147 146 148 155 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 167 167 167 167 130 132 132 132 132 132 132 133 132 133 132 133 133 133 132 132 131 135 136 140 139 140 142 142 142 143 143 143 143 143 143 143 143 144 145 145 146 147 147 147 147 147 147 147 147 147 147 147 147 148 148 148 148 148 149 149 149 150 152 156 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 167 168 169 169 132 132 132 132 132 132 132 132 132 132 132 132 134 132 130 134 134 135 133 143 142 142 142 142 142 143 143 143 143 143 143 143 142 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 147 148 148 148 148 148 150 152 153 155 158 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 167 170 172 172 133 132 132 132 132 132 132 133 133 133 133 133 135 131 128 133 132 133 133 143 142 142 142 142 142 143 143 143 143 143 143 143 143 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 147 148 148 148 148 148 150 152 151 153 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 166 170 172 172 131 132 132 132 132 132 132 133 133 133 133 133 134 136 134 137 137 138 137 143 142 142 142 142 142 143 143 143 143 143 143 143 146 147 147 147 147 147 147 147 147 147 147 147 147 148 150 150 150 150 150 149 148 148 150 153 156 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 159 160 160 161 162 162 164 165 165 166 167 167 167 167 167 167 169 171 172 172 172 129 132 132 132 132 132 132 133 133 133 133 133 134 142 140 141 143 144 143 143 143 143 143 143 142 143 143 143 143 143 143 145 150 151 148 148 148 147 147 147 147 147 148 148 147 150 153 153 153 153 152 151 149 149 150 154 161 160 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 163 163 166 168 168 167 167 167 167 167 167 168 171 177 175 173 173 129 132 132 132 132 132 132 132 132 132 132 132 133 141 139 141 142 143 143 142 142 142 142 142 142 143 143 142 143 143 143 145 150 151 148 148 148 147 147 147 147 147 148 148 147 150 153 153 153 151 152 153 154 154 153 154 160 160 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 162 162 165 168 167 167 167 167 167 167 167 168 171 176 175 173 173 130 132 132 132 132 132 132 133 133 134 135 135 137 143 136 138 143 143 143 143 143 143 143 143 143 143 143 144 145 145 145 145 148 149 147 147 147 147 147 147 147 147 148 148 147 149 151 150 152 155 154 155 157 157 156 155 159 159 157 157 157 157 157 157 157 157 157 158 158 158 158 158 159 160 160 162 163 163 163 165 165 166 167 167 168 167 167 168 169 169 170 172 175 174 173 173 132 132 132 132 132 132 132 133 132 134 138 138 137 135 138 141 142 142 142 143 143 143 143 143 143 142 142 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 148 148 148 147 147 147 152 158 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 168 168 168 168 168 167 167 167 170 172 172 173 173 173 173 173 173 132 132 132 132 132 132 132 132 132 133 137 137 136 135 139 142 144 144 143 143 143 143 143 142 142 142 142 144 148 147 148 147 147 147 147 147 147 147 147 147 147 147 147 148 148 147 147 146 152 158 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 162 163 163 163 163 165 167 167 167 167 167 167 167 167 170 172 172 173 173 173 173 173 173 132 133 133 133 133 133 133 133 132 135 136 138 139 141 136 136 136 139 138 140 140 140 143 149 142 144 144 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 148 150 150 150 151 151 153 156 156 156 158 158 157 155 156 157 159 159 158 157 157 157 157 157 156 157 157 157 157 157 162 166 165 165 165 165 166 167 167 167 167 167 169 170 170 171 172 172 173 173 173 173 173 173 132 132 132 132 132 132 133 132 131 136 134 139 136 137 137 135 134 132 149 156 157 154 148 145 148 156 155 154 152 152 153 154 153 154 154 153 154 154 153 155 156 156 155 153 152 156 162 161 158 154 154 157 160 160 158 156 156 155 155 154 160 166 166 166 166 166 166 166 166 167 166 166 167 168 168 168 168 167 169 169 169 169 169 169 172 173 173 174 175 175 174 173 173 174 175 175 132 132 132 132 132 132 133 132 131 136 134 139 136 138 138 136 137 144 106 77 74 79 128 133 136 135 135 136 139 139 143 148 147 148 148 147 148 148 146 150 156 155 156 153 153 154 156 155 157 158 158 160 163 163 164 164 165 163 162 161 164 167 165 164 163 162 161 159 158 158 158 160 163 165 167 169 169 168 173 177 176 176 176 176 175 172 172 177 180 180 175 171 171 176 180 179 132 133 132 133 133 132 132 132 131 134 133 139 139 135 135 136 131 147 41 39 77 77 122 132 129 127 124 124 122 119 117 124 122 125 126 124 121 119 122 126 130 130 123 128 126 129 132 133 132 132 132 134 135 135 138 142 140 143 146 145 147 154 164 169 174 184 191 198 202 205 204 193 186 180 170 163 162 162 164 167 166 168 169 168 170 171 170 177 182 181 178 176 176 181 185 184 132 133 133 133 133 133 133 132 132 132 134 139 138 132 133 130 124 129 44 103 172 155 149 148 150 151 150 148 147 148 151 153 151 145 154 165 163 161 159 161 152 159 158 163 170 159 166 155 161 154 155 154 154 154 153 151 152 150 148 148 148 151 158 160 164 171 171 185 205 213 207 189 176 169 163 155 150 151 150 150 150 144 141 141 142 142 142 143 144 144 145 146 145 147 149 148 132 133 133 133 133 133 133 132 131 135 141 136 133 132 133 130 124 130 41 9 20 35 60 80 84 114 152 186 211 214 175 90 150 164 140 89 89 90 97 95 65 72 66 51 52 62 69 59 109 168 159 162 161 161 160 158 158 160 162 161 162 162 159 159 161 157 156 156 151 152 153 149 156 165 167 166 163 164 164 164 164 166 167 167 165 166 165 167 169 169 165 162 163 162 160 160 133 132 132 132 132 132 132 132 132 134 138 134 131 132 133 132 132 133 91 33 61 91 88 119 122 135 175 213 222 217 122 14 103 125 88 96 98 81 110 99 102 66 38 15 25 87 65 78 100 166 154 154 157 157 156 159 159 158 157 158 157 157 161 160 159 162 164 164 164 164 165 167 167 165 162 161 162 161 161 162 161 165 168 168 165 163 163 168 172 171 174 177 177 175 173 174 133 132 132 132 132 132 132 132 132 132 131 132 133 133 133 132 132 129 124 50 66 96 99 133 150 158 177 202 186 141 44 12 55 53 65 113 108 97 116 105 113 92 90 55 70 107 118 90 120 168 156 154 160 160 154 156 157 157 157 157 160 163 163 163 164 163 163 163 163 163 163 163 162 162 162 163 163 163 162 163 162 165 167 166 169 170 170 171 172 172 172 172 172 174 175 175 132 132 132 132 132 132 132 132 132 132 132 132 132 132 132 131 131 131 111 63 84 111 136 164 170 169 166 151 113 49 18 12 63 85 103 120 103 106 111 120 116 117 95 59 96 118 113 84 117 169 157 154 157 162 161 160 159 160 161 159 161 164 163 162 161 161 161 161 160 161 161 161 162 161 162 162 161 162 162 162 162 166 170 168 173 179 178 176 176 175 174 174 174 178 181 181 132 132 132 132 132 132 132 132 132 132 132 132 132 132 132 132 129 128 104 106 146 164 173 179 151 101 76 53 93 90 42 23 89 72 77 113 98 77 97 76 93 87 72 77 61 85 102 117 122 160 154 162 169 149 124 138 140 138 138 151 152 155 162 166 172 173 170 171 174 171 173 169 172 176 168 166 168 165 162 166 162 164 159 161 169 157 161 162 154 166 166 162 161 173 177 175 132 132 132 132 132 132 132 132 132 132 132 132 133 133 133 134 127 127 100 170 163 141 122 110 88 32 52 56 49 76 85 84 64 64 87 108 59 84 93 72 98 112 116 116 114 95 71 65 102 155 137 149 139 59 36 62 79 81 96 73 84 87 87 98 126 136 155 167 173 166 165 167 173 180 168 162 164 159 147 145 141 136 105 97 92 76 98 99 75 98 100 104 78 124 194 167 132 132 132 132 132 132 132 132 132 132 132 132 133 133 133 134 127 125 107 174 139 98 90 108 120 74 88 114 97 126 126 132 117 74 81 46 31 80 88 115 123 94 95 105 110 115 119 99 111 156 149 114 30 44 32 52 75 69 84 53 58 61 59 67 92 101 129 135 136 154 158 150 151 158 155 150 147 144 124 117 114 113 84 108 106 71 95 134 99 113 116 112 96 88 152 172 132 133 133 133 133 133 133 132 132 132 132 132 133 132 132 132 131 124 118 165 124 96 101 95 78 83 97 103 92 71 75 89 54 36 10 63 66 39 23 108 106 52 76 93 101 79 101 118 122 158 164 123 55 81 92 118 114 67 49 52 62 81 98 103 111 106 120 121 130 161 163 152 136 145 145 137 132 132 122 113 108 109 86 108 118 85 90 127 93 115 114 98 99 93 86 168 133 133 133 133 133 133 133 132 132 132 132 132 133 133 133 133 136 127 118 156 124 130 88 57 52 65 46 63 54 108 116 108 102 69 23 28 28 67 75 78 78 93 132 110 89 53 90 73 94 163 163 125 73 133 133 126 109 67 66 56 81 89 108 128 128 121 124 124 151 170 157 163 140 131 138 156 155 139 131 127 124 119 101 117 104 97 122 122 69 96 112 97 93 91 98 175 132 133 133 133 133 133 133 132 132 132 132 132 132 133 133 133 136 126 120 151 94 73 35 82 129 137 97 26 14 58 63 44 124 87 85 61 99 123 108 96 73 89 53 49 34 40 74 44 68 163 173 122 64 123 121 111 95 67 79 68 83 83 102 129 136 128 135 124 128 151 148 150 121 121 148 167 169 142 125 134 139 134 127 144 122 108 122 120 69 74 103 86 84 79 89 174 132 133 133 133 133 133 134 135 135 135 135 135 134 132 132 132 135 126 115 155 95 48 45 115 135 126 130 51 15 10 26 68 71 87 120 122 114 100 107 91 16 35 33 24 70 106 102 53 54 163 170 121 65 101 98 110 95 63 68 65 94 120 136 140 147 142 138 138 158 161 140 154 145 143 155 153 168 158 132 131 140 145 142 145 133 124 132 115 82 71 81 72 68 69 80 173 132 133 133 133 133 133 134 137 137 137 137 137 136 132 132 132 135 125 120 137 73 37 64 96 107 108 81 60 16 17 44 66 70 118 118 68 65 114 96 93 31 32 35 39 100 121 139 91 46 165 164 122 76 99 88 99 102 86 79 90 113 130 144 150 155 154 170 184 175 173 162 162 177 172 167 155 158 158 139 139 153 155 142 138 131 126 139 110 103 103 111 81 89 66 81 172 132 133 132 133 133 132 134 137 137 137 136 137 136 133 133 133 136 126 119 128 86 40 90 122 125 119 108 99 29 56 44 62 113 97 67 43 90 77 22 55 57 18 68 99 54 99 98 58 55 162 165 129 86 102 87 102 115 88 86 108 126 146 159 167 162 171 183 174 173 172 170 160 162 171 176 174 167 163 155 162 153 144 142 139 133 123 125 121 124 117 90 92 117 83 89 173 132 133 137 136 133 139 136 139 136 138 140 138 133 135 131 130 134 126 124 109 74 52 90 133 101 91 130 117 66 127 138 90 17 26 107 133 122 108 39 13 31 43 103 131 121 132 107 99 66 163 164 128 93 101 84 111 103 95 137 143 133 153 169 171 165 184 191 181 178 169 167 168 164 165 169 171 163 189 193 192 185 163 145 138 136 124 110 98 128 162 115 84 68 81 90 176 132 133 140 138 133 144 137 141 136 139 142 140 132 136 130 127 132 126 129 116 67 41 89 120 91 85 117 106 71 121 73 37 8 90 133 131 124 144 79 20 18 56 120 123 123 117 129 108 62 167 159 130 106 114 84 79 98 138 169 168 143 157 172 166 162 185 187 183 179 174 170 171 171 171 170 175 164 180 186 187 196 194 174 151 131 117 104 85 127 149 150 123 74 78 119 171 132 133 139 138 133 143 136 140 135 135 133 142 138 135 130 127 132 126 130 111 72 76 99 115 117 123 117 54 24 94 101 38 34 59 70 117 84 99 106 13 48 116 114 130 106 76 127 59 70 171 149 134 113 119 92 88 125 162 163 177 163 166 173 163 157 180 182 177 172 172 175 175 175 175 176 175 168 164 182 184 187 193 195 169 140 129 117 103 135 159 154 144 96 75 115 171 132 134 139 139 135 140 138 142 140 140 140 142 139 135 130 127 132 127 128 125 121 132 128 108 81 93 85 27 14 46 40 13 95 116 110 121 86 95 114 20 32 119 102 107 125 118 75 35 98 162 153 129 96 105 89 87 97 139 159 168 166 167 174 165 153 176 181 174 168 175 183 182 181 182 176 177 173 151 176 175 187 180 191 179 150 128 109 96 106 142 148 135 102 93 95 177 132 134 139 140 137 137 139 143 143 145 147 141 138 135 130 127 132 126 128 129 104 65 104 142 107 106 110 33 18 53 22 16 87 136 120 116 134 127 126 62 60 128 116 63 73 96 42 76 67 158 159 127 84 98 86 64 83 145 160 167 177 179 177 155 146 179 180 177 172 169 168 168 176 167 163 177 186 180 186 177 168 190 197 188 161 132 111 98 89 147 153 163 149 111 98 172 132 133 139 140 136 137 139 143 143 145 147 141 138 135 130 127 132 127 130 130 93 73 107 120 126 132 89 25 19 54 28 20 71 130 93 89 116 128 99 49 110 120 119 127 50 73 122 104 50 163 158 126 80 92 87 103 131 166 163 173 188 196 194 176 150 175 189 183 177 173 170 171 172 164 170 177 165 181 176 172 178 190 192 182 154 134 121 109 124 167 167 163 167 153 111 167 134 139 141 142 140 140 141 143 143 144 145 142 141 138 133 130 131 125 131 136 135 125 106 116 135 131 88 35 24 9 17 25 40 123 111 101 128 131 47 23 94 95 91 128 119 62 88 65 59 154 157 125 78 85 132 158 151 158 168 176 183 192 202 202 173 171 184 186 194 174 173 173 164 179 190 170 150 170 173 169 187 171 164 165 157 138 125 121 169 183 169 162 161 166 138 167 135 144 143 143 143 143 142 143 143 143 143 143 143 140 135 132 131 124 131 137 131 121 119 122 131 129 122 84 88 71 26 13 17 63 107 119 104 56 24 36 74 133 90 105 133 69 90 119 77 146 155 125 54 97 167 148 149 158 165 175 177 188 200 196 188 177 183 179 190 196 186 181 185 189 173 146 147 152 153 145 144 150 151 157 160 139 125 149 174 173 169 161 163 157 139 168 135 143 143 143 142 142 142 143 143 143 143 143 143 140 135 132 131 124 126 143 128 121 119 128 125 130 136 100 88 131 74 44 81 90 82 89 102 39 16 21 75 134 113 69 134 103 60 120 104 143 156 124 34 116 164 147 151 157 160 169 176 190 201 195 196 184 186 174 172 200 203 189 181 182 170 149 146 145 147 143 144 147 147 153 151 134 127 168 175 174 164 161 162 158 141 168 138 141 140 140 143 143 143 143 143 143 143 143 143 140 135 133 133 125 134 147 120 114 123 125 119 130 130 86 51 61 81 116 103 121 132 126 133 97 37 39 96 127 123 97 123 117 48 102 101 146 153 125 38 131 155 156 144 148 154 165 173 186 195 196 203 194 195 188 189 183 187 189 184 193 187 163 152 160 157 149 150 150 152 156 152 139 135 169 180 179 169 166 163 161 122 165 137 136 138 140 143 142 143 143 143 143 143 143 143 140 135 133 134 125 145 144 125 112 98 115 115 119 130 125 81 55 124 124 121 125 121 121 125 126 121 122 118 128 120 124 121 121 36 84 111 149 152 131 45 131 154 156 151 144 149 158 168 185 193 196 207 200 203 198 200 195 187 181 181 197 199 178 155 157 157 154 154 150 153 151 155 150 143 175 181 174 171 166 162 156 102 171 135 130 141 147 143 143 143 143 143 143 143 143 143 140 135 133 134 123 150 137 112 116 111 99 105 116 133 129 120 126 119 124 126 119 123 123 124 118 121 121 112 130 120 113 110 127 48 81 129 150 156 129 64 129 156 153 147 154 151 153 152 159 166 177 214 204 198 197 197 196 193 188 186 198 196 183 163 156 158 161 161 158 160 161 153 141 134 173 182 177 170 161 156 154 104 169 134 134 139 142 140 140 140 143 143 143 143 144 141 139 139 134 137 127 161 135 97 83 77 77 70 124 128 122 122 126 125 130 132 125 122 123 120 115 124 102 110 114 102 106 119 135 56 96 124 148 154 120 83 147 154 134 94 121 140 140 117 122 137 168 209 198 192 196 199 199 192 187 183 195 201 183 178 164 173 181 175 164 161 164 161 154 155 174 172 160 151 151 152 159 121 166 134 138 137 137 137 137 139 143 143 143 143 144 140 137 141 134 135 125 191 149 78 54 45 25 96 119 117 128 126 125 125 123 123 123 121 121 122 127 132 98 92 109 118 121 119 126 87 120 115 147 151 119 92 163 143 100 64 76 76 75 80 97 112 145 195 194 178 181 187 194 189 185 178 184 197 176 175 156 163 165 160 159 166 181 196 184 169 170 165 150 145 149 152 155 119 171 133 137 137 137 138 138 139 143 143 143 143 144 140 138 141 135 133 120 216 156 79 67 64 63 123 106 115 126 125 128 125 121 121 122 125 126 128 125 119 104 115 113 119 112 116 120 126 127 119 148 151 122 71 143 141 82 70 77 68 67 78 86 93 122 165 182 184 177 184 192 191 186 184 180 194 184 174 145 166 171 174 187 192 207 205 191 192 188 178 176 165 157 152 153 98 175 135 141 141 139 134 134 137 143 143 143 143 143 142 140 138 130 133 113 210 162 78 60 59 108 125 117 119 108 97 89 90 98 99 103 102 95 90 105 93 79 87 88 106 122 119 118 123 117 118 146 149 130 15 67 148 54 62 58 59 64 74 97 85 125 154 170 185 186 189 194 188 177 184 182 191 185 175 148 163 172 178 203 198 199 199 194 190 182 174 172 167 158 156 151 85 172 137 143 143 140 135 135 137 143 143 143 143 143 142 141 136 130 133 114 191 154 91 67 70 114 123 111 118 118 109 118 111 108 113 114 112 107 99 98 94 110 110 105 111 117 114 120 117 118 116 146 151 122 26 57 132 61 91 89 65 54 77 97 112 131 151 163 167 177 181 190 185 174 183 182 187 179 171 152 167 188 187 208 201 203 199 195 190 181 174 168 164 161 162 136 80 172 141 143 143 142 141 141 141 143 143 143 143 143 142 141 136 133 134 123 182 163 105 80 113 115 126 110 111 116 112 100 119 108 112 108 113 121 126 112 107 111 109 119 124 119 118 120 119 118 117 147 149 130 29 46 41 41 96 96 92 94 96 99 74 92 122 133 143 157 170 183 180 180 184 176 183 175 174 161 175 200 189 203 203 203 200 194 193 185 176 172 171 169 143 72 53 182 142 139 140 139 139 139 141 142 142 142 142 142 143 141 136 134 134 125 158 156 107 94 120 114 124 94 91 100 99 46 46 79 67 45 48 77 95 101 97 87 89 93 104 102 100 119 114 115 114 142 150 122 14 29 63 29 45 69 57 71 85 93 105 124 133 143 141 147 159 174 185 185 178 167 173 175 175 174 180 191 183 189 196 197 195 194 170 158 157 152 141 132 102 51 57 184 141 137 136 139 146 146 145 143 143 143 143 143 143 141 136 134 135 126 141 149 145 135 113 103 95 56 79 115 90 72 58 83 114 105 83 103 103 104 106 103 100 99 88 37 86 117 111 112 112 139 149 119 10 47 56 24 53 64 83 78 52 52 67 75 96 116 115 119 138 148 166 186 172 162 171 173 192 200 186 181 168 142 168 191 184 176 129 105 102 98 79 65 42 16 40 188 141 137 136 139 144 143 144 142 142 142 142 142 142 140 135 133 134 125 144 160 150 125 124 111 90 50 71 117 103 115 122 126 117 123 125 128 122 120 119 117 112 96 70 9 72 124 113 114 113 141 146 135 117 95 117 103 95 116 115 106 109 108 98 94 112 129 132 138 145 149 159 165 151 166 181 175 182 169 149 146 135 115 137 154 155 139 97 92 92 87 76 82 79 77 103 170 140 140 140 140 139 140 140 143 143 143 143 143 143 141 136 132 133 127 130 148 147 124 115 113 84 33 84 102 93 111 102 99 96 86 80 83 78 89 95 101 99 94 48 5 55 115 101 105 103 139 142 141 144 153 148 152 153 151 155 156 164 170 170 175 183 186 193 196 192 194 195 180 162 157 169 170 163 164 170 171 166 160 156 156 161 151 152 170 168 155 155 164 165 178 188 174 139 143 143 143 143 144 143 142 143 143 143 142 143 141 137 132 132 128 125 148 147 129 119 110 104 86 111 114 102 108 100 95 85 66 70 77 68 61 49 28 26 31 27 20 36 106 90 101 101 140 143 142 140 146 150 150 149 152 156 154 157 161 161 163 165 170 173 174 175 179 183 175 159 152 158 166 166 165 172 175 171 167 165 166 161 157 167 175 181 161 152 158 153 157 164 172 139 143 143 143 143 142 143 143 143 143 143 143 143 141 137 131 131 128 126 148 150 124 109 132 143 133 123 128 118 127 115 108 117 113 116 113 103 102 85 56 44 44 91 76 34 119 115 103 100 146 148 147 147 153 156 156 156 158 161 161 162 163 163 161 162 168 168 166 166 167 169 179 108 94 104 97 98 98 100 105 111 110 106 105 105 97 124 181 171 156 130 119 110 106 110 115 138 142 142 142 143 142 142 143 143 143 143 143 142 138 137 134 134 126 121 147 141 89 63 82 105 132 129 117 113 115 100 95 97 89 102 94 103 105 103 107 83 67 108 74 20 97 105 104 101 151 158 154 154 156 158 157 157 158 159 159 159 160 160 159 159 160 162 162 162 164 164 169 112 109 112 109 110 110 108 108 112 110 109 112 116 122 123 139 135 135 134 129 116 100 104 104 136 142 142 142 143 143 142 143 143 143 143 143 141 137 138 136 136 126 118 146 143 95 81 71 74 91 112 118 105 87 88 87 84 84 100 87 90 83 87 96 88 80 97 76 26 97 101 100 97 153 162 155 158 157 157 158 158 158 158 158 158 158 158 158 158 158 161 163 163 164 162 161 114 110 110 111 111 111 112 113 112 110 113 119 124 130 136 134 137 141 138 137 129 114 98 93 137 143 142 143 142 143 143 143 143 143 143 143 141 137 138 136 135 126 118 147 162 117 87 84 95 89 84 111 110 102 104 97 87 82 82 83 85 89 83 77 85 93 89 77 27 91 97 94 95 153 162 156 157 158 158 158 158 158 158 158 158 158 158 158 158 158 161 163 163 161 157 156 113 115 115 115 116 116 113 111 113 116 118 122 129 134 138 146 146 140 134 138 137 129 116 103 138 143 142 143 143 143 143 143 143 143 143 143 141 137 138 136 133 131 112 150 169 113 88 101 110 105 95 83 115 111 97 98 87 83 87 90 83 82 86 84 87 87 79 72 33 57 88 86 85 154 163 155 158 158 158 158 158 158 158 158 158 158 158 158 158 158 159 160 160 159 156 149 119 121 121 119 118 118 116 118 104 102 119 129 129 136 143 141 139 136 135 139 138 134 136 122 139 143 143 143 143 143 143 143 143 143 143 143 141 137 137 136 131 134 109 140 155 116 105 103 109 107 106 100 88 122 122 111 101 90 100 103 91 96 95 94 92 97 101 100 102 88 88 86 86 159 163 156 158 157 157 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 156 147 122 122 122 119 117 117 116 116 68 81 118 128 132 142 144 138 138 138 138 137 137 138 140 130 139 143 143 143 143 142 142 143 143 143 143 143 141 137 137 136 131 131 116 131 117 94 87 88 101 109 113 121 97 107 143 119 97 99 144 150 110 86 90 90 78 76 93 93 94 90 93 79 88 168 160 159 158 157 157 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 156 147 121 122 122 119 117 115 126 101 34 96 132 143 146 153 145 137 138 137 137 137 138 134 132 123 140 137 140 143 143 143 143 143 143 143 143 143 141 137 138 136 132 131 120 119 73 71 75 82 92 97 99 104 118 112 126 138 117 127 157 154 138 103 108 109 102 97 104 107 111 107 106 101 121 164 158 159 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 159 155 146 122 122 123 122 121 120 132 76 84 153 138 141 145 140 141 139 137 137 136 136 137 125 106 101 132 133 139 143 143 143 143 143 143 143 143 143 141 137 138 137 132 133 113 134 135 135 131 136 138 136 136 138 142 141 143 150 144 132 136 148 152 152 152 152 153 154 152 150 149 150 155 162 163 157 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 160 154 145 123 123 123 122 123 123 119 108 123 79 91 137 146 146 143 129 137 146 144 148 124 118 87 80 135 134 139 143 143 143 143 143 143 143 143 143 141 137 137 136 131 132 114 142 141 137 143 148 152 151 151 149 147 147 146 150 152 142 140 148 150 148 148 148 147 147 149 150 150 149 151 156 157 156 155 155 156 156 156 156 156 156 156 155 157 158 158 158 158 158 158 158 158 160 154 145 123 123 123 122 123 128 126 70 15 18 49 70 84 110 130 140 144 138 118 119 162 130 82 77 140 134 139 143 143 143 143 143 143 143 143 143 142 140 136 134 132 132 116 145 143 147 147 148 148 146 146 146 146 146 149 150 155 155 161 158 159 160 160 161 163 163 162 161 162 161 149 160 164 170 166 170 164 164 168 172 171 169 171 172 165 160 161 162 158 158 158 158 158 159 156 143 126 119 132 122 125 92 48 13 25 93 121 101 71 48 43 71 101 124 137 80 73 148 118 55 139 134 139 143 143 143 143 143 143 143 143 143 143 141 136 133 133 133 116 145 145 147 148 148 148 148 148 148 148 147 151 155 150 155 145 147 142 135 136 135 133 135 129 122 124 119 151 168 168 122 119 119 124 122 131 133 133 132 138 136 140 171 157 157 158 158 158 158 158 158 157 142 123 120 127 109 43 10 17 21 52 121 113 120 119 104 76 45 31 43 89 143 85 44 136 107 139 133 139 143 143 143 143 143 143 143 143 143 143 141 136 133 133 133 116 145 145 147 148 148 148 147 147 147 147 146 153 138 41 29 19 22 19 14 15 14 12 13 11 10 10 10 91 153 93 11 8 8 27 28 13 21 19 16 20 12 56 175 164 159 157 158 158 158 158 158 157 142 122 129 85 19 17 22 21 17 46 80 73 80 105 131 138 100 56 41 39 51 105 81 54 131 144 139 142 143 143 142 143 143 143 143 143 143 142 138 137 135 132 132 115 144 144 146 147 147 147 148 148 148 148 146 154 138 20 12 12 12 12 12 12 13 16 15 17 15 17 23 18 25 15 24 23 19 18 18 18 25 39 31 28 26 41 172 166 158 157 158 158 158 158 158 157 141 133 78 6 16 23 24 21 19 41 64 67 65 85 112 103 103 78 32 41 43 32 74 57 47 142 143 143 143 143 143 143 143 143 143 143 143 141 137 137 136 131 132 115 144 144 146 147 147 147 148 148 148 148 147 152 136 22 15 13 13 13 13 13 13 13 12 14 14 19 24 21 19 24 23 17 14 12 12 14 22 37 33 25 23 37 169 162 157 158 158 158 158 158 158 157 145 112 17 22 21 24 22 25 25 30 46 55 65 55 45 27 38 45 43 30 33 29 33 81 64 141 142 142 142 142 142 142 143 143 143 143 143 141 137 137 136 131 132 115 144 144 146 147 147 147 147 147 147 147 146 151 130 15 11 13 13 13 13 13 13 13 13 10 17 25 17 15 15 17 18 11 10 14 12 16 23 25 33 31 28 39 157 162 159 158 158 158 158 158 158 156 150 79 6 27 19 23 23 18 26 28 23 44 96 45 22 47 32 31 23 43 54 38 34 56 121 140 143 143 143 143 143 143 143 143 143 143 143 141 137 138 135 131 130 116 143 147 147 147 147 147 147 147 148 149 145 146 126 16 9 10 10 10 10 10 10 11 15 14 14 17 13 13 13 14 14 12 12 13 13 15 21 24 31 27 26 32 150 160 157 158 158 158 158 158 159 159 148 79 11 20 21 23 23 20 26 33 38 29 111 125 54 86 105 87 100 115 90 59 54 39 64 143 143 143 143 143 142 143 143 143 143 143 143 141 137 138 134 131 129 117 142 148 147 147 147 147 148 148 150 151 145 145 126 18 8 9 8 8 8 8 8 10 16 16 13 12 13 13 13 13 13 13 13 13 14 9 11 28 25 13 24 30 148 159 156 158 158 158 158 158 159 162 144 94 17 18 21 22 21 29 22 28 40 38 118 156 132 101 105 138 144 137 101 61 59 55 58 142 143 143 143 143 143 143 143 143 143 143 143 141 137 138 134 131 129 117 142 148 147 147 147 147 148 148 150 151 145 145 126 18 8 9 8 8 8 8 8 10 16 16 13 13 13 13 13 14 14 13 13 13 13 13 15 13 28 34 26 29 149 159 156 158 158 158 158 158 159 162 142 113 32 18 25 24 23 22 31 37 44 87 140 131 138 147 146 138 134 140 122 72 68 59 52 140 143 143 143 143 143 143 143 143 143 143 143 141 137 137 135 128 130 116 142 148 147 147 147 147 147 147 148 148 146 146 129 19 8 8 8 8 8 8 9 12 14 14 12 13 8 10 16 9 16 17 11 13 12 14 15 8 17 32 27 25 144 162 156 158 158 158 158 158 159 163 140 97 61 19 16 16 22 27 15 60 121 116 139 140 137 122 136 141 139 129 128 85 56 36 28 138 142 142 142 142 142 142 143 143 143 143 143 141 137 137 136 127 131 116 143 149 148 148 148 148 147 147 147 147 147 147 132 19 7 8 8 8 8 8 10 13 13 13 12 13 9 9 19 11 11 20 16 12 13 13 12 14 12 21 27 24 142 164 156 158 157 157 158 158 159 163 139 85 84 40 9 11 12 20 14 41 110 107 140 129 117 115 103 130 130 130 120 88 61 35 29 138 142 142 142 142 142 142 143 143 143 143 143 141 136 137 136 127 131 116 143 149 148 148 148 148 147 147 147 147 147 147 130 18 8 7 8 7 8 8 10 13 13 13 12 12 16 14 16 29 15 5 11 13 13 12 13 13 13 23 27 24 142 164 156 157 158 158 158 158 159 163 139 88 75 78 38 20 18 15 19 18 55 65 53 41 73 144 142 121 123 129 126 100 65 36 29 138 142 142 142 142 142 142 143 143 143 143 143 143 143 140 135 128 134 112 136 146 147 147 147 147 147 147 147 147 147 147 135 21 8 9 13 12 7 8 10 13 13 13 13 14 7 29 97 120 108 51 17 8 13 14 14 12 15 24 24 20 141 161 155 158 158 158 158 158 159 160 137 78 71 77 63 42 26 16 18 22 24 27 10 44 119 126 118 114 127 132 127 108 55 29 31 138 142 142 142 142 142 142 143 143 143 143 143 144 146 141 135 128 135 110 133 145 147 147 147 147 147 147 147 147 147 147 137 22 8 10 16 14 7 8 10 13 13 13 13 14 8 33 108 117 122 129 115 80 19 10 5 20 23 19 19 16 141 159 155 158 158 158 158 158 159 159 136 72 72 66 60 49 33 16 10 15 20 23 22 38 70 83 101 112 127 127 129 121 52 29 25 138 142 142 142 142 142 142 143 143 143 143 143 144 146 141 135 128 135 111 134 146 148 148 148 148 147 147 147 147 147 147 136 22 8 10 16 14 7 8 10 13 13 12 13 12 13 30 90 115 115 125 126 140 85 10 13 14 14 15 14 14 142 160 155 158 157 157 158 158 159 159 136 71 66 62 56 50 43 28 18 13 18 25 21 30 78 122 144 133 126 119 134 108 38 23 43 141 142 142 142 142 142 142 143 143 143 143 143 143 144 142 139 130 135 106 137 146 147 148 148 147 147 147 147 147 147 147 137 22 8 9 10 10 8 8 10 13 13 13 12 13 11 15 82 119 115 119 126 124 124 33 2 16 10 13 12 12 138 162 154 158 157 157 157 158 154 155 135 65 63 62 54 44 36 33 30 25 19 17 24 55 99 111 128 131 124 130 121 51 43 44 36 142 142 142 142 142 142 142 143 143 143 143 143 142 142 143 140 131 135 104 138 147 147 148 148 148 147 147 147 147 147 147 137 22 8 8 8 8 8 8 10 13 13 13 12 13 13 21 76 119 124 120 119 121 126 52 0 19 14 12 13 12 137 164 154 158 157 157 158 158 154 156 132 59 58 59 58 51 41 34 29 25 20 17 18 90 137 129 138 133 131 116 51 57 97 99 90 142 143 143 142 142 142 142 143 143 143 142 143 143 143 143 140 131 135 104 138 147 147 148 148 148 147 147 147 147 147 147 137 22 8 8 8 8 8 7 10 13 13 12 13 14 11 16 28 79 122 127 121 127 133 89 9 7 11 13 12 12 138 163 154 158 158 158 158 157 159 161 128 54 54 54 53 47 38 33 30 24 25 21 19 34 63 47 107 91 59 24 53 114 115 131 134 138 139 139 140 142 142 142 143 143 143 143 143 143 142 138 135 132 135 104 138 147 147 148 148 148 147 147 147 147 147 148 141 29 9 8 8 8 8 8 9 9 9 12 10 11 18 26 8 18 77 108 93 96 106 92 37 8 7 15 12 14 139 168 154 158 158 157 157 157 156 155 127 45 50 56 55 46 34 28 28 26 25 22 22 17 20 17 14 15 13 40 104 128 132 134 128 136 135 135 137 142 142 142 142 142 142 142 142 142 141 136 133 132 135 104 138 147 147 148 148 148 147 147 147 147 147 148 142 32 9 8 8 8 8 8 8 7 7 11 4 0 43 94 7 4 87 88 10 34 90 93 41 16 14 14 16 27 140 165 154 158 158 158 157 157 155 153 127 43 48 51 51 46 39 30 26 28 24 21 23 18 20 19 20 26 42 74 115 127 134 132 133 138 139 139 140 142 142 142 143 143 143 143 143 143 141 136 133 132 135 104 138 147 147 148 148 148 147 147 147 147 147 148 142 31 9 8 8 8 7 7 7 7 7 8 39 38 48 84 26 10 134 75 12 56 79 115 61 1 17 11 19 46 139 156 154 159 157 158 158 158 156 154 128 45 47 40 40 43 43 34 26 28 24 22 23 17 19 19 17 17 49 98 132 130 132 133 133 138 139 139 140 142 142 142 142 142 142 142 142 141 138 137 135 131 135 102 136 151 148 148 148 148 147 147 147 147 147 147 142 32 7 9 8 8 7 7 7 7 7 6 49 117 128 91 20 36 130 118 78 111 121 122 68 3 19 34 47 52 136 159 155 158 157 157 158 158 152 157 131 40 49 46 39 38 40 33 27 28 29 21 19 26 30 19 38 32 80 126 130 129 126 125 125
## 4264                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   48 75 129 191 201 185 186 173 159 174 185 172 165 172 251 254 217 222 175 181 182 169 180 194 191 206 234 234 232 178 187 204 189 166 133 77 36 22 8 2 12 16 7 12 25 23 27 49 27 8 154 194 127 161 148 119 124 125 121 135 130 111 172 244 214 185 182 207 222 208 101 34 32 33 34 35 36 37 38 38 39 41 44 42 38 29 18 10 6 3 2 3 9 10 35 78 42 82 152 191 186 178 175 175 168 177 202 205 201 200 253 247 236 226 156 145 172 156 157 168 178 197 226 251 243 233 215 198 215 233 213 213 180 157 133 109 97 82 66 62 50 49 15 19 2 20 143 200 134 143 109 103 136 139 133 147 143 133 194 234 214 205 194 181 198 202 84 19 27 30 35 34 41 47 34 29 24 47 58 35 10 17 73 144 179 178 171 161 129 87 67 61 43 90 169 198 188 179 177 180 175 182 212 217 203 199 239 240 239 237 192 187 152 138 149 144 124 130 172 206 200 204 177 133 157 168 150 198 177 179 193 205 212 194 167 167 181 208 136 142 115 119 180 187 123 149 121 113 146 147 138 151 147 138 193 222 211 212 203 180 187 196 92 28 35 38 38 36 32 31 37 49 47 43 13 16 90 179 225 236 241 241 232 235 235 215 184 131 38 88 179 201 192 185 188 190 175 169 190 197 186 181 210 225 234 243 232 235 206 205 218 209 165 136 152 169 172 203 200 192 209 190 154 184 163 115 137 182 188 167 160 183 200 229 215 216 141 136 202 187 120 163 145 136 165 165 153 164 157 148 199 219 207 207 206 192 200 202 96 38 42 28 23 34 42 33 25 26 40 16 52 138 211 241 241 227 234 240 218 207 209 213 216 200 50 96 190 201 194 191 196 200 180 163 177 192 206 198 210 228 228 226 228 224 209 197 196 214 210 195 203 208 218 233 228 242 232 229 225 222 211 193 208 225 218 205 203 208 201 223 228 222 97 89 195 191 121 169 160 149 176 174 159 169 160 155 200 226 214 207 199 194 208 212 104 28 30 40 35 19 21 29 26 40 11 57 155 225 237 232 240 245 243 240 239 238 234 223 215 204 63 103 181 194 193 194 201 209 193 177 192 206 206 200 213 243 239 230 236 226 227 203 171 181 193 201 221 220 227 231 235 237 174 181 197 194 216 230 224 214 223 237 223 186 164 206 223 216 104 95 182 182 126 165 156 145 169 167 151 162 154 152 192 223 228 218 196 182 199 210 112 27 52 120 127 72 70 97 55 35 50 154 228 242 245 245 241 238 235 230 226 216 199 183 172 170 66 103 160 181 192 199 206 216 203 183 194 206 204 197 219 253 246 233 237 221 221 216 188 171 159 169 206 213 202 158 191 175 113 124 126 141 175 203 208 197 188 178 157 139 137 182 199 187 149 155 179 177 145 164 150 140 162 162 145 159 151 155 189 211 228 224 202 187 200 204 104 31 78 156 144 72 112 192 126 48 160 224 250 248 248 229 196 178 170 167 175 176 172 166 153 145 70 105 159 174 193 203 205 213 199 177 182 192 194 184 216 252 246 239 242 228 198 208 210 199 180 184 216 223 213 172 189 159 138 159 144 158 166 164 190 209 191 155 146 163 160 167 181 162 186 207 194 190 165 169 154 146 166 168 151 166 159 167 195 203 217 216 209 207 215 208 104 41 94 155 116 49 114 220 178 89 236 231 230 220 194 177 171 161 151 150 142 122 103 89 73 67 73 106 175 179 195 198 191 196 191 179 192 204 197 180 218 252 245 247 250 242 182 168 176 201 222 231 233 217 215 219 205 203 212 222 200 207 205 194 203 219 218 204 203 215 195 185 204 190 224 237 222 210 166 176 165 152 172 173 156 173 167 176 202 208 211 207 210 214 219 220 135 51 75 120 77 39 102 198 217 172 237 195 184 172 145 154 177 161 127 108 85 52 30 26 18 16 72 91 158 177 194 191 181 186 189 190 209 220 201 178 220 255 245 242 244 243 193 151 133 168 225 249 232 198 190 215 199 211 241 239 220 219 222 239 232 225 229 231 229 223 216 212 227 213 216 213 224 215 154 172 168 154 177 178 158 178 169 175 198 217 209 207 205 204 208 227 188 100 93 130 90 64 124 202 233 225 206 153 162 157 156 174 161 104 51 32 28 20 23 43 38 30 78 78 122 172 197 193 189 194 195 197 209 215 194 165 207 254 246 230 226 227 227 201 166 175 220 247 237 217 205 222 222 196 224 206 193 193 190 211 215 217 214 202 199 201 217 208 206 185 165 157 198 211 149 166 167 156 181 181 158 185 174 170 190 224 203 204 200 195 194 218 212 178 190 220 182 135 175 223 198 183 176 141 178 172 169 152 95 38 20 26 34 34 35 55 40 31 85 87 132 181 203 199 198 203 200 198 206 216 211 171 202 249 248 234 220 212 242 246 229 230 245 249 246 249 249 247 249 223 203 139 137 170 170 172 168 157 160 163 161 176 183 189 189 186 182 166 193 209 156 176 179 165 184 181 159 191 183 177 195 228 200 195 200 196 185 194 163 180 231 248 229 186 189 203 170 146 158 164 176 158 129 61 21 31 40 34 37 33 36 60 40 31 93 92 133 174 193 193 193 196 192 188 191 199 210 173 200 245 253 250 245 244 237 225 214 223 237 234 227 233 237 236 232 208 193 141 154 190 167 156 139 114 128 157 162 174 158 169 200 204 208 183 189 202 161 187 192 183 201 195 168 193 186 182 203 215 204 201 211 207 193 184 111 123 195 226 222 183 163 170 163 151 158 169 146 96 51 21 34 39 29 28 28 30 50 86 55 29 97 91 126 168 196 202 201 199 198 198 194 191 207 187 218 248 245 240 235 240 195 146 115 129 170 193 203 213 203 166 207 235 242 191 165 175 151 149 141 138 167 192 169 131 103 124 199 171 165 161 190 219 176 185 182 178 203 199 168 182 176 175 206 219 222 212 201 196 200 200 112 102 163 188 184 161 162 174 164 148 183 141 62 15 25 32 46 44 38 36 25 27 61 108 71 33 96 89 125 163 198 214 211 204 205 211 202 182 188 190 226 248 244 242 237 239 237 204 168 173 204 227 231 217 178 142 213 250 251 227 191 189 170 170 167 183 201 195 154 105 132 181 196 163 177 185 199 210 160 166 158 155 181 181 155 171 168 165 197 218 221 215 201 199 207 212 141 119 162 185 177 169 178 172 156 155 202 84 26 18 34 40 44 42 39 35 29 43 82 118 69 31 92 91 131 155 189 210 209 198 197 205 194 180 215 216 229 226 223 235 233 230 242 248 250 253 255 250 230 183 134 176 203 193 201 245 253 255 210 162 146 173 203 222 222 189 210 224 161 163 224 244 226 215 175 197 181 162 180 178 161 183 183 167 191 214 200 209 212 216 211 206 170 151 170 180 171 171 170 147 150 181 158 29 36 49 31 34 48 31 23 30 27 43 66 78 40 38 89 89 132 165 194 215 221 210 199 199 192 176 195 201 209 192 201 239 252 254 216 206 218 235 240 236 235 219 185 195 202 197 200 223 229 244 208 172 154 142 137 159 196 209 208 202 188 171 218 235 225 225 176 172 176 179 193 189 172 192 193 172 196 232 201 206 196 204 202 202 206 204 193 171 152 160 163 150 155 158 87 14 16 29 49 40 23 33 37 23 35 69 89 78 36 44 88 84 125 171 197 216 228 217 197 192 187 180 206 230 242 214 209 241 249 249 250 230 217 233 246 242 243 250 243 247 231 230 212 169 149 215 238 174 158 144 112 99 116 153 146 150 178 153 191 201 200 226 194 190 172 165 186 183 163 177 182 168 200 222 203 222 204 214 215 202 206 186 162 162 161 159 142 143 165 145 77 30 40 44 47 52 68 81 110 119 78 58 38 22 25 94 97 89 118 164 179 205 219 213 191 172 171 176 195 177 219 249 235 247 249 227 241 244 239 236 236 233 232 231 225 234 250 234 247 216 176 211 217 231 202 129 112 155 169 153 167 182 200 186 206 198 205 237 199 184 179 173 184 179 160 175 171 177 207 227 211 209 208 207 212 208 189 175 171 162 153 160 161 157 145 95 28 32 40 39 43 51 72 112 169 172 85 17 22 39 25 48 108 100 132 167 183 201 207 206 188 167 165 169 182 162 205 244 245 253 243 206 187 188 210 238 244 225 217 229 236 239 240 228 254 241 217 228 193 198 171 110 97 133 161 176 167 98 95 99 145 136 164 226 194 193 191 185 198 198 182 196 186 189 211 220 207 210 209 206 205 192 173 163 160 153 148 153 151 143 118 54 25 35 38 32 39 55 83 128 171 152 71 28 43 50 15 14 100 93 134 192 224 227 222 227 216 195 189 188 201 192 219 227 217 230 237 229 217 199 198 228 247 236 227 234 236 249 237 238 255 239 236 239 188 108 105 140 166 168 152 156 139 52 64 80 137 115 149 224 192 201 193 177 192 194 178 189 173 171 192 219 206 214 210 207 200 176 165 159 152 149 148 149 143 136 108 36 32 41 36 29 39 67 108 154 195 172 86 54 60 51 20 42 88 82 133 194 237 229 216 224 219 198 187 179 178 181 209 221 223 231 228 228 208 192 182 202 235 246 244 248 245 243 230 235 229 216 236 247 217 152 146 167 183 185 163 144 131 88 117 137 173 139 164 224 184 196 192 177 190 190 173 184 170 170 193 226 209 216 210 211 199 169 167 165 154 149 149 143 134 133 105 34 38 47 37 29 43 82 136 182 221 221 165 107 52 34 62 120 98 88 145 190 234 224 207 217 216 194 178 183 203 192 199 196 194 195 192 206 156 151 151 174 224 250 244 238 238 239 235 229 202 200 229 240 237 222 195 162 162 188 191 177 181 175 166 187 197 174 194 225 187 197 195 183 196 196 181 197 188 192 214 229 210 212 211 215 200 168 168 168 155 148 144 133 123 121 88 24 39 50 36 30 45 87 151 196 212 219 216 175 115 90 120 168 117 96 152 182 222 219 204 209 210 188 171 189 222 212 210 189 164 153 154 179 114 113 115 146 214 249 237 234 241 238 248 241 210 212 224 226 242 242 213 185 193 213 206 196 210 210 193 214 209 204 220 230 195 198 193 183 196 197 185 203 194 196 216 220 206 204 213 217 195 165 161 158 147 142 139 128 120 112 68 16 38 49 34 31 41 80 149 195 205 205 207 200 192 161 141 173 122 90 145 176 210 218 208 204 205 187 172 199 226 221 220 189 154 140 133 140 113 127 128 152 202 216 207 230 244 217 238 244 233 232 223 220 241 245 229 216 223 227 213 206 212 203 217 230 225 218 222 228 194 192 189 180 193 195 182 198 185 184 202 210 203 199 217 213 182 160 150 142 136 136 135 129 129 112 58 21 39 48 33 33 38 69 141 192 205 202 196 203 214 184 149 162 116 80 139 176 204 222 216 204 204 190 178 216 237 230 221 175 138 144 144 137 146 175 179 191 204 190 186 219 222 195 203 211 231 231 214 210 214 233 241 235 231 234 228 211 206 215 237 238 242 216 206 226 196 194 191 183 195 195 181 196 184 183 201 208 209 198 220 205 166 157 145 134 132 132 130 127 133 110 49 28 41 49 34 37 38 64 139 196 208 204 206 208 191 173 178 178 121 83 148 172 199 218 213 202 203 195 184 211 234 241 228 166 131 152 166 176 167 184 199 217 225 205 196 209 193 198 189 180 222 231 203 190 160 172 199 206 207 221 222 198 211 237 248 240 242 196 196 233 205 207 197 186 199 198 187 205 192 190 205 214 218 198 217 205 162 152 147 133 126 126 124 116 119 102 46 26 38 45 43 41 38 58 137 201 215 209 204 197 179 178 200 196 133 89 159 165 199 211 204 204 207 203 192 190 225 244 228 162 142 158 159 177 146 151 171 194 210 194 175 188 181 186 189 168 218 224 186 178 135 118 112 101 107 143 173 185 240 230 214 229 230 177 198 233 201 206 197 191 205 202 193 211 193 186 197 217 221 198 207 211 174 147 151 136 113 114 122 107 94 93 57 24 29 41 54 42 39 42 118 198 213 199 193 195 213 206 169 142 131 89 155 165 198 210 206 206 206 200 189 184 200 213 222 163 125 162 181 180 138 124 109 112 149 170 167 179 166 127 126 130 161 135 119 157 154 144 115 89 88 111 128 131 167 147 84 174 249 219 215 221 186 199 198 192 200 191 178 195 181 179 194 216 217 207 196 200 187 155 146 137 114 105 118 112 90 74 48 28 23 56 40 39 56 32 79 186 225 206 210 217 219 177 117 102 137 93 153 167 199 211 211 215 214 204 191 178 171 175 206 154 100 161 204 197 144 107 69 87 177 230 189 133 101 119 131 145 155 128 111 136 161 145 125 98 71 90 111 56 19 30 17 77 225 250 240 230 195 191 188 190 204 201 192 209 190 182 193 223 217 212 192 206 212 166 126 123 115 100 102 108 101 63 35 46 30 51 32 29 61 44 52 157 229 220 215 202 194 181 171 184 128 79 135 166 201 212 217 227 230 216 196 180 177 188 233 177 96 121 134 117 127 123 96 106 174 201 142 98 110 180 204 197 185 194 164 139 170 154 95 64 57 58 65 52 32 29 41 57 171 217 221 231 213 221 211 203 210 200 189 210 199 202 218 224 222 214 198 208 215 181 142 139 126 106 118 113 81 42 35 23 42 28 38 18 46 74 62 119 203 227 217 199 196 197 190 177 135 85 146 166 196 207 215 233 241 225 197 170 156 176 227 172 95 100 97 102 112 116 91 78 125 173 167 172 193 173 170 137 113 132 110 76 105 145 92 63 65 34 14 36 49 47 85 77 63 56 161 236 202 202 201 198 205 196 186 208 199 204 219 207 225 212 204 197 191 198 203 147 113 118 90 67 82 66 55 46 40 18 49 20 26 82 93 97 159 214 207 202 206 192 163 152 117 72 142 157 189 201 207 226 240 227 194 171 183 196 233 160 83 86 84 115 95 110 110 81 89 127 141 141 143 150 115 88 71 61 50 47 65 67 55 65 71 63 62 46 32 79 93 99 94 80 175 236 199 211 201 192 207 206 200 219 201 195 204 203 218 206 215 207 190 207 222 232 184 102 67 67 62 20 37 44 25 32 48 33 21 56 106 105 95 124 164 195 216 217 217 234 124 85 152 155 190 204 204 219 234 226 196 169 175 188 237 164 81 92 87 105 110 115 105 78 88 132 146 115 80 110 91 86 99 94 87 101 120 113 68 65 97 101 84 70 60 85 99 75 89 99 183 242 205 193 200 206 213 201 190 211 200 204 219 216 209 210 221 217 203 200 200 214 221 181 96 34 49 47 57 31 28 42 38 34 32 27 82 122 111 117 151 155 147 154 188 226 123 86 145 155 196 212 207 215 230 226 200 173 172 172 218 134 42 72 80 88 83 100 122 108 81 74 77 79 66 66 86 89 130 171 171 187 223 195 90 51 95 109 83 82 78 70 96 98 79 43 133 224 206 207 199 194 208 204 197 216 199 196 207 221 205 221 220 207 199 190 193 200 221 234 182 97 52 15 43 33 45 39 29 26 42 18 53 132 147 131 157 165 158 141 128 133 132 92 162 168 198 213 211 221 235 228 207 193 193 197 221 152 59 82 81 74 75 74 80 77 74 71 58 60 75 87 77 47 156 190 162 178 225 231 146 89 80 87 121 129 83 57 47 52 65 65 159 237 213 209 201 199 213 207 196 211 197 198 209 218 225 222 213 202 182 175 198 230 227 198 191 197 162 85 55 53 42 35 44 37 32 24 51 132 167 143 147 161 153 141 101 74 133 94 171 177 196 213 218 223 231 228 209 186 174 205 239 177 64 79 82 48 50 63 79 87 93 90 76 63 50 104 85 47 182 202 158 182 230 230 144 95 91 94 119 126 99 95 79 63 58 66 174 242 200 190 181 183 202 196 179 186 175 188 201 219 223 216 208 203 193 190 201 160 125 127 118 110 99 80 69 51 35 30 43 38 31 26 46 125 175 156 149 169 178 172 120 95 139 91 168 184 197 216 224 222 223 223 206 175 163 191 224 179 72 92 101 57 56 74 81 74 68 69 82 88 51 98 100 58 188 207 180 213 230 213 133 100 110 105 106 112 97 89 68 79 128 139 195 235 198 193 185 187 206 203 184 187 178 197 199 225 217 208 207 215 216 200 178 112 83 96 65 32 29 44 49 28 30 29 38 34 35 25 36 114 173 163 151 167 187 182 125 109 150 84 152 180 197 215 221 219 220 220 205 184 181 183 219 219 156 156 127 82 67 76 74 61 58 71 103 134 109 116 126 71 192 214 201 229 200 194 132 94 103 107 112 118 104 96 97 86 188 221 215 228 210 202 199 200 212 211 198 205 194 208 188 217 216 205 204 217 210 176 141 117 114 102 66 42 35 27 27 28 31 32 36 32 38 25 27 102 166 165 154 158 173 166 117 109 167 88 146 164 193 207 207 215 224 220 208 206 206 187 224 251 233 198 109 82 79 76 67 62 75 92 100 123 129 119 120 66 207 237 222 241 198 200 145 93 102 130 140 127 108 121 124 60 173 244 226 225 215 210 208 204 209 207 200 213 198 213 179 200 219 202 188 195 176 142 133 145 144 115 90 86 66 25 20 43 33 35 35 30 40 26 24 95 163 168 158 157 165 158 119 119 173 94 149 153 190 202 197 213 227 216 207 219 215 193 211 235 233 174 66 75 91 75 58 65 103 125 100 95 116 96 104 66 217 250 232 245 212 216 174 119 120 149 151 120 101 129 117 80 185 243 232 224 205 212 209 201 206 201 194 207 188 217 193 205 222 187 163 175 163 144 162 181 163 126 115 125 91 25 16 44 33 33 35 30 39 28 27 93 162 169 154 158 166 157 122 127 152 84 142 150 190 204 199 213 224 208 198 213 218 203 196 224 219 141 54 83 87 64 48 66 123 152 114 87 91 71 110 89 212 243 234 245 217 222 199 146 120 127 130 122 117 133 118 148 223 215 212 225 200 210 200 192 202 197 190 199 171 217 214 221 214 168 150 177 184 175 182 196 172 131 135 154 108 31 18 41 30 29 35 31 36 29 30 88 157 167 148 158 162 145 111 113 130 76 139 150 184 203 206 215 219 204 192 206 228 210 186 248 242 136 66 77 64 52 44 64 118 144 117 86 61 72 129 118 214 242 241 246 224 221 197 140 104 106 124 153 156 147 161 181 205 157 175 228 214 212 191 184 199 197 192 200 160 211 219 207 188 163 163 185 199 184 154 155 148 131 151 162 104 36 31 42 30 27 39 34 33 27 29 77 147 169 154 167 154 129 102 96 135 80 150 153 175 204 215 219 221 207 190 201 223 202 170 243 235 117 58 47 42 48 51 66 101 123 125 123 90 111 141 122 215 252 241 236 223 219 179 118 106 137 161 174 171 157 217 174 180 146 166 227 219 207 186 181 195 195 196 208 167 211 221 171 166 169 182 185 189 165 107 88 104 132 166 151 76 29 39 33 33 30 36 37 37 26 24 62 132 171 169 178 156 114 101 99 144 68 137 159 178 215 222 223 229 206 176 187 180 191 183 211 183 92 57 41 48 68 86 93 103 113 127 152 140 125 104 74 174 248 240 246 229 227 174 109 116 171 181 150 169 181 184 109 149 159 169 215 206 189 177 168 175 179 183 202 185 220 237 165 174 171 185 183 187 150 78 41 70 138 185 147 58 28 43 26 36 41 26 38 50 31 21 41 87 141 171 173 168 90 82 115 151 86 118 157 175 205 210 211 219 198 172 189 189 181 179 216 170 106 112 85 98 146 150 124 136 157 141 140 152 138 115 98 194 249 240 252 225 219 179 154 174 182 145 123 163 173 144 90 134 200 233 225 185 172 167 159 168 178 181 201 190 226 237 204 173 188 192 192 192 108 30 10 84 189 205 139 58 29 49 35 26 51 50 41 31 28 31 24 21 74 152 177 177 112 78 126 144 105 145 166 177 206 217 221 225 201 176 188 192 203 179 192 173 127 140 141 158 179 177 158 154 157 146 139 138 158 140 128 214 249 242 252 218 225 206 196 200 185 155 143 160 155 171 185 213 230 239 228 202 196 194 184 190 193 183 190 175 201 197 222 183 199 204 193 150 49 18 28 114 203 155 64 25 27 45 27 49 49 35 27 28 47 66 48 19 36 100 155 176 138 84 107 120 82 155 161 171 208 230 238 235 208 182 195 209 192 196 171 103 115 179 188 196 169 151 143 118 99 113 127 112 133 135 119 194 239 238 251 220 221 218 202 175 163 182 187 170 158 221 205 214 206 213 224 208 204 195 178 187 195 186 194 185 210 193 215 193 153 159 176 127 36 19 33 136 254 200 70 20 35 43 26 56 34 17 41 74 96 111 110 102 114 127 100 70 124 128 68 141 71 130 145 160 197 221 228 227 213 199 208 214 207 239 176 74 90 129 139 139 127 121 124 121 106 92 87 104 142 163 128 183 244 239 241 221 203 196 189 162 144 163 184 199 202 210 187 229 204 198 231 207 183 190 191 198 203 191 197 184 202 177 192 199 147 134 175 154 55 23 15 91 213 194 71 21 46 42 36 24 53 67 104 145 151 143 144 152 142 119 100 69 79 92 49 183 99 130 155 168 193 206 211 218 222 223 219 196 214 243 179 103 91 85 107 87 86 102 129 152 154 132 116 143 171 204 159 199 250 245 242 231 222 202 199 194 167 147 172 230 241 205 194 251 212 193 236 216 192 187 182 190 197 193 208 194 209 189 186 158 145 146 187 184 76 27 25 37 66 64 22 13 28 18 49 51 121 143 157 183 182 163 147 161 163 134 146 165 125 65 38 168 93 127 166 180 195 201 202 214 224 223 213 189 167 198 193 120 115 153 160 136 114 117 138 148 157 175 177 180 160 195 165 208 251 241 250 246 248 234 226 226 211 187 198 246 239 227 177 191 187 204 228 203 200 208 203 204 201 195 207 180 182 169 198 114 103 130 168 199 125 38 26 34 25 29 33 46 58 64 114 149 192 200 197 192 175 163 161 154 158 168 154 171 224 196 150 133 63 112 162 180 193 198 199 207 213 199 184 186 171 175 217 192 151 174 179 187 172 163 165 157 161 180 173 168 173 202 183 226 245 225 244 238 240 247 242 235 237 239 235 246 220 236 234 220 193 208 221 195 188 193 189 192 191 192 213 184 188 180 189 136 118 129 127 191 202 94 22 30 37 20 24 85 157 192 208 230 228 234 240 199 146 146 185 175 145 181 228 209 217 227 230 116 60 113 178 189 188 200 208 218 225 215 205 199 193 190 218 167 113 143 136 138 139 150 173 189 183 169 175 181 172 203 209 245 252 235 243 232 222 238 249 240 235 240 242 239 230 246 253 212 177 205 232 209 186 174 173 188 187 183 204 174 207 207 95 180 142 123 121 168 204 151 73 40 29 35 106 183 212 229 232 249 243 247 242 179 150 171 185 190 188 195 241 231 216 222 245 124 69 114 177 193 196 212 218 221 225 220 217 214 208 194 228 208 158 169 153 134 114 117 154 196 194 167 175 182 164 188 197 230 248 244 245 225 214 227 241 238 228 232 247 247 239 248 255 227 199 222 236 212 195 182 178 193 191 185 205 175 214 190 41 97 90 151 163 122 133 170 123 48 42 107 202 238 232 244 248 250 243 249 242 172 160 196 201 205 213 208 235 233 222 214 230 130 74 106 169 193 201 218 222 222 222 217 216 214 207 178 225 247 205 185 166 163 162 168 180 194 194 185 189 183 203 211 202 208 219 230 243 235 223 211 212 221 222 225 240 249 236 223 229 230 219 236 234 210 206 203 192 195 189 180 199 176 205 135 28 39 38 104 131 78 69 107 91 41 69 176 243 239 234 247 248 249 237 242 240 183 180 220 224 219 221 216 225 224 224 202 203 138 85 107 166 188 191 203 210 216 219 212 216 224 215 180 223 253 204 168 152 161 179 200 210 206 197 197 213 211 230 235 230 216 215 228 243 247 238 213 203 216 230 236 239 249 240 212 206 222 220 238 231 207 204 209 197 192 188 183 206 194 215 101 28 52 41 36 58 46 44 49 49 42 98 211 243 221 233 240 248 241 234 227 231 203 213 241 236 228 233 226 223 224 229 202 193 144 96 114 169 190 187 193 203 215 220 214 217 219 208 189 231 254 209 188 182 188 196 214 231 235 221 217 237 239 227 230 242 233 231 238 234 236 240 234 228 230 240 247 235 239 243 223 200 220 212 228 227 204 194 204 197 196 198 192 209 206 217 83 32 70 67 25 32 35 40 43 52 55 111 215 232 209 233 235 247 220 236 218 222 228 242 244 231 230 237 230 227 230 230 212 200 141 98 113 172 202 203 206 212 222 225 222 223 202 189 192 238 251 230 234 232 233 238 240 228 222 227 238 249 238 226 214 233 229 230 233 219 228 237 246 250 240 237 231 196 200 229 226 195 225 205 215 221 207 192 204 203 200 196 176 176 175 172 53 37 73 71 35 42 44 44 44 57 57 100 200 223 204 234 237 217 194 241 223 224 244 251 233 221 224 225 218 220 228 217 214 204 138 98 112 166 206 213 215 216 221 221 221 227 205 190 200 240 249 241 248 244 237 243 219 155 118 135 174 219 235 237 206 230 227 220 214 196 226 238 243 248 237 227 208 157 167 213 223 187 234 209 214 218 209 193 206 203 189 173 146 142 146 122 31 40 78 61 33 42 50 48 38 43 44 85 180 217 205 240 239 166 180 244 235 235 247 245 223 216 217 210 205 208 222 206 209 204 148 107 117 161 199 205 206 210 216 216 214 217 192 188 202 238 255 254 254 252 251 242 194 121 65 34 48 152 245 225 186 223 238 233 211 174 206 227 235 242 228 216 212 186 197 217 217 171 232 216 223 223 208 185 196 188 167 156 151 169 176 111 30 39 85 57 41 39 42 46 50 38 33 76 156 215 223 235 225 143 188 243 242 244 241 237 223 216 212 207 202 202 221 207 207 203 156 109 112 168 201 199 198 208 223 226 221 214 187 198 203 225 239 229 223 221 213 214 219 214 179 100 43 110 205 155 110 145 179 215 239 214 230 220 222 236 217 201 226 245 244 215 176 135 209 210 222 222 207 184 189 183 164 163 181 211 212 102 18 48 91 59 42 34 41 46 42 28 29 58 120 212 239 208 190 147 206 237 238 244 235 232 227 218 208 205 201 198 216 207 202 199 157 107 93 166 191 184 186 202 218 225 222 215 224 232 182 168 158 113 101 93 68 60 74 71 50 36 44 76 121 128 93 55 20 71 196 244 217 223 226 223 218 216 228 232 218 201 136 145 226 223 204 201 209 207 200 204 195 185 187 192 193 82 31 74 76 65 44 36 42 47 44 39 42 40 82 206 236 181 172 142 221 224 228 232 233 230 225 223 222 222 222 218 215 214 216 213 145 99 77 141 193 204 208 223 232 231 238 248 234 179 72 37 46 27 20 12 32 57 73 64 53 64 95 128 147 121 120 82 57 16 55 184 240 253 243 218 217 239 223 167 153 203 176 195 220 171 170 190 206 225 234 218 177 156 174 182 159 48 28 71 72 64 47 37 38 43 44 48 43 26 81 213 207 166 173 169 235 239 239 236 231 230 234 239 247 249 239 226 219 218 220 216 124 111 94 162 216 218 207 211 204 178 171 162 114 163 160 156 170 170 165 137 155 171 151 124 121 137 148 146 135 117 96 104 84 52 20 40 150 228 232 207 216 230 197 172 175 189 207 210 217 171 167 175 170 181 193 192 160 136 150 161 132 32 38 68 66 60 50 38 34 37 43 53 47 26 88 223 187 174 190 211 254 250 245 243 247 253 254 251 241 236 248 253 242 228 219 218 127 130 108 202 208 145 89 70 59 43 36 32 23 63 62 69 117 156 155 110 89 116 132 151 167 152 120 98 96 115 107 145 71 72 89 20 33 183 248 195 178 212 198 156 157 201 172 163 194 166 136 146 148 136 121 146 162 141 130 135 120 41 54 67 58 52 49 42 34 34 40 48 52 42 93 217 183 189 184 228 254 244 234 236 246 251 244 230 224 231 243 253 255 253 247 235 116 138 116 163 141 90 61 56 61 75 87 79 64 53 28 28 51 78 111 129 101 97 77 71 83 91 104 121 131 123 123 129 93 49 56 59 25 120 213 193 167 193 203 189 184 189 120 136 185 176 159 179 168 142 124 146 164 143 124 122 105 42 63 66 52 46 45 44 37 34 37 41 53 48 79 198 192 178 141 221 243 250 253 253 242 220 203 198 219 231 202 181 188 218 244 248 122 156 135 125 111 107 112 100 78 77 83 69 56 69 78 72 46 31 53 85 83 80 52 42 61 102 142 129 105 119 119 99 134 99 48 66 48 60 175 233 205 182 200 232 201 127 151 143 151 175 207 205 148 128 159 167 142 126 125 119 93 42 73 63 49 42 43 45 40 36 34 38 53 44 59 175 215 168 115 235 248 248 237 201 147 111 129 183 227 226 183 129 90 89 126 178 164 172 126 110 108 113 119 102 69 56 64 65 65 56 57 66 76 85 78 64 51 61 58 45 40 74 118 82 38 79 116 104 122 138 104 74 52 49 160 215 179 167 191 181 142 135 228 188 158 180 212 203 153 165 185 157 121 113 117 105 85 49 72 57 50 44 43 43 41 37 34 42 58 43 42 142 223 160 115 248 254 222 179 124 68 48 100 191 245 241 211 154 89 51 68 124 175 161 96 113 123 119 118 111 84 58 55 61 57 37 46 64 68 60 49 45 50 56 53 43 42 102 181 156 90 35 74 72 84 104 89 72 63 45 79 89 82 115 149 142 166 213 174 173 178 171 151 164 159 184 166 128 117 118 103 81 80 58 57 52 51 48 44 42 40 38 35 46 63 48 29 101 206 146 113 236 247 216 195 187 168 149 167 218 250 246 223 191 166 160 170 183 114 111 115 111 125 112 112 127 117 90 81 86 84 71 67 71 68 51 38 43 47 42 48 51 46 118 235 208 115 26 59 47 41 65 45 47 37 47 40 29 49 102 154 175 207 239 195 179 193 146 116 157 156 141 134 127 124 126 112 87 80 55 57 49 52 50 45 40 38 37 35 48 62 49 25 62 177 168 128 224 251 236 230 226 216 205 208 229 246 243 230 217 209 205 208 215 76 111 123 113 123 113 117 138 141 130 122 106 92 95 91 81 69 47 36 44 48 45 43 37 47 124 197 157 127 32 40 45 29 35 54 120 77 37 38 78 124 168 212 229 227 219 238 228 226 190 146 130 120 111 127 129 121 126 109 66 58 53 59 47 50 49 43 38 36 36 35 47 57 50 33 50 158 201 164 212 252 246 247 242 233 230 235 244 249 246 237 224 221 227 231 229 60 122 113 118 113 112 120 128 118 110 106 72 53 66 69 60 51 38 31 37 41 41 38 30 56 123 147 111 128 42 28 58 38 13 58 196 181 115 113 152 168 169 194 223 233 222 245 252 236 238 196 120 92 93 109 113 116 129 112 61 51 58 54 44 48 47 40 34 33 35 35 47 55 51 38 41 129 217 203 200 246 247 253 253 244 243 250 250 247 246 243 227 224 240 245 235 96 162 109 112 102 110 120 129 118 107 94 42 32 45 45 36 37 36 33 32 39 40 33 41 79 109 87 68 94 53 28 43 39 24 36 161 228 208 191 198 181 144 134 153 186 205 231 241 236 250 227 152 92 62 62 80 97 107 102 76 63 60 50 45 49 47 39 33 32 34 35 49 56 49 37 34 97 210 229 213 242 248 253 248 234 236 248 247 242 244 242 232 229 238 239 232 163 222 112 102 109 119 114 143 177 179 132 42 45 61 58 41 36 37 36 38 41 41 31 53 106 106 46 39 67 58 35 21 31 38 16 65 190 228 207 202 200 176 149 140 168 198 216 209 218 229 225 192 129 65 34 44 59 64 73 76 63 51 43 49 53 50 42 34 31 33 35 54 61 46 36 39 74 164 203 214 239 247 246 238 223 226 237 226 220 220 207 215 229 231 232 232 136 187 112 101 125 124 94 143 224 240 166 43 40 62 65 46 34 32 34 40 37 35 44 67 110 123 84 72 83 104 110 101 81 24 2 10 129 220 204 180 185 191 194 197 214 219 193 168 163 186 198 205 189 133 70 35 33 52 68 65 49 39 32 51 56 54 45 35 32 34 36 57 64 44 36 47 53 80 110 150 235 248 235 232 232 233 230 198 186 183 163 189 225 227 224 227 24 68 111 109 123 112 83 134 206 217 161 56 26 43 46 38 38 37 32 32 40 45 65 64 87 129 119 89 101 172 226 252 203 62 49 60 120 196 184 158 157 159 169 178 181 167 164 170 162 173 166 180 212 190 136 61 34 63 69 46 34 37 30 49 55 55 46 36 33 34 37 54 57 35 32 53 49 24 32 68 236 253 226 221 237 246 244 217 205 199 186 213 242 232 212 212 32 45 113 125 116 95 81 113 137 128 104 54 44 51 47 42 47 43 34 30 29 38 45 29 65 131 127 98 103 91 175 221 236 175 180 196 221 178 119 109 123 126 124 132 154 161 174 225 222 151 115 154 195 189 169 91 41 50 36 21 29 38 37 43 50 52 45 36 33 34 37 52 52 29 27 57 66 35 33 32 218 250 222 212 227 231 234 233 230 229 227 232 241 238 209 202 156 111 108 110 93 81 80 106 116 112 104 61 37 49 49 41 35 29 29 38 34 40 49 42 65 124 161 146 84 35 97 97 147 165 151 173 225 177 86 45 58 81 98 119 160 192 200 214 167 41 60 183 220 204 163 88 47 44 28 35 49 35 34 36 43 48 43 35 32 35 38 60 64 34 26 49 66 46 44 5 152 238 237 229 227 198 174 159 156 169 172 169 186 216 209 204 147 115 93 100 93 114 108 120 137 135 116 53 26 52 61 49 37 29 25 30 38 44 41 42 60 73 63 44 32 9 21 49 53 9 31 135 210 174 136 128 112 85 73 76 84 109 119 61 10 71 197 246 223 200 113 37 36 64 70 60 59 52 65 32 26 39 44 33 31 39 42 56 55 35 42 63 63 54 47 14 73 237 245 226 237 227 213 181 152 149 159 165 175 207 215 222 163 155 145 155 127 109 88 111 131 117 98 63 52 57 49 33 32 38 41 43 52 56 48 40 47 56 52 47 58 81 37 6 5 14 72 152 180 146 138 150 130 84 62 72 91 101 16 38 103 195 244 231 170 115 48 9 20 46 54 55 59 49 58 46 39 45 48 39 31 34 40 57 57 32 28 55 67 48 41 42 34 204 231 227 244 242 236 214 203 216 220 216 204 212 224 218 145 159 157 156 145 136 112 125 123 87 64 44 33 38 38 39 50 67 78 82 60 55 53 48 50 57 63 71 87 108 100 106 126 133 151 164 132 107 102 103 92 80 91 114 125 138 162 130 131 128 91 69 62 41 63 79 80 73 47 34 35 27 38 37 32 29 35 35 24 21 35 63 63 37 39 63 61 38 38 43 30 205 244 234 237 228 223 205 188 190 199 204 199 192 206 214 111 127 123 106 92 81 64 92 122 114 89 50 26 41 57 62 56 53 53 48 63 76 84 91 98 104 111 118 128 141 150 163 177 162 157 158 128 90 61 41 36 45 60 69 72 77 76 37 19 11 7 34 81 99 117 133 137 117 79 60 56 43 49 40 37 30 36 43 32 27 44 49 50 53 45 51 58 46 37 28 6 189 247 237 235 234 242 233 212 204 208 216 219 194 182 205 111 117 109 80 67 79 79 94 118 130 125 84 47 58 80 87 76 73 73 66 65 70 83 95 103 106 108 108 111 131 127 130 147 140 117 91 49 35 31 29 31 34 33 39 61 81 79 91 104 112 118 132 148 154 155 152 139 111 79 67 63 50 49 43 39 31 37 45 34 28 46 52 49 41 32 59 76 38 26 61 8 184 245 234 225 224 230 220 217 231 230 222 222 199 171 195 133 129 125 133 139 165 159 129 102 98 107 107 128 116 117 114 97 98 103 91 79 75 83 94 101 104 106 104 96 95 87 80 95 97 92 97 86 76 78 78 77 82 89 103 122 126 135 158 161 153 155 148 137 136 110 81 66 59 63 70 69 59 52 41 33 28 35 40 33 30 45 61 51 26 27 49 51 41 46 40 38 181 234 230 228 228 230 221 208 211 211 197 195 197 185 208 151 137 143 141 145 157 147 131 126 137 152 148 142 125 126 127 112 119 131 123 101 89 90 98 101 103 109 108 101 112 118 109 104 94 92 109 113 113 109 92 79 82 89 93 94 77 75 90 86 76 73 63 48 45 43 31 30 54 79 72 49 38 34 56 44 42 47 44 38 38 44 46 41 35 33 39 60 102 100 25 149 231 247 242 242 231 222 216 207 204 205 198 194 202 204 224 159 137 152 140 161 169 145 134 145 153 151 140 133 128 143 148 123 120 130 121 131 126 124 128 127 125 130 129 119 122 128 114 108 114 124 134 120 104 98 88 84 88 80 76 91 96 113 113 103 78 49 30 27 31 34 23 24 58 80 57 37 59 83 76 60 57 54 39 29 28 25 34 44 34 39 109 186 175 136 136 229 253 232 231 241 228 218 223 215 195 193 203 199 193 194 214 145 149 155 166 175 167 158 168 177 175 171 164 154 152 162 157 143 148 147 135 145 140 135 138 140 137 134 134 129 117 134 116 98 121 128 131 114 108 107 106 109 111 102 94 101 110 125 99 78 66 48 32 30 40 42 30 34 66 82 64 56 84 84 47 44 43 33 24 62 121 142 107 53 37 98 200 231 177 154 190 246 251 234 228 230 218 210 215 210 191 189 203 205 197 203 218 103 123 102 135 164 156 144 143 131 120 131 147 137 143 149 145 143 149 150 146 144 136 128 122 120 119 119 123 122 112 127 141 133 150 135 117 104 117 121 117 117 115 110 104 101 103 107 86 73 73 59 33 22 28 26 23 26 46 78 86 89 73 45 42 49 39 34 65 151 220 213 139 82 101 180 247 234 179 146 140 214 239 244 230 218 212 208 207 205 198 200 212 214 207 214 226 118 135 90 126 168 172 161 145 105 58 46 61 63 80 80 84 84 73 78 81 76 80 86 89 96 114 135 147 147 142 132 127 144 143 111 129 127 132 130 122 114 107 100 94 93 95 92 88 84 75 50 22 15 25 36 56 82 98 95 80 72 49 101 149 154 132 127 164 213 217 169 120 136 194 232 246 241 208 150 105 158 212 244 240 232 230 223 212 212 217 216 217 216 214 218 223 195 212 172 197 211 213 220 218 183 119 51 5 13 36 14 16 19 11 35 54 74 98 121 137 152 177 200 210 207 205 164 66 87 69 47 171 195 190 185 179 172 165 159 155 153 153 145 146 143 130 112 102 109 122 141 163 197 205 135 43 28 37 197 251 241 237 230 211 206 198 155 122 166 232 237 225 235 234 191 152 130 161 188 208 231 242 235 221 226 239 232 219 212 208 190 165 237 253 244 245 241 242 247 248 247 225 144 56 78 98 53 46 63 79 133 166 207 231 241 242 238 239 241 242 240 245 176 34 42 17 21 207 250 249 250 252 254 255 255 255 254 253 245 251 254 249 245 246 249 251 251 250 250 241 162 28 41 94 227 229 219 245 248 200 200 231 207 164 184 230 231 221 237 247 234 224 155 130 112 130 169 197 207 215 224 230 221 192 169 154 119 84 226 236 239 229 228 233 240 238 239 243 214 177 214 219 168 162 194 214 244 243 238 238 234 227 220 214 212 216 228 241 160 43 51 27 34 202 250 239 236 239 242 243 244 244 244 243 241 248 250 246 243 245 245 243 244 249 249 244 179 54 96 225 250 205 213 228 244 234 231 247 246 238 245 253 251 250 250 248 247 253 206 170 130 114 117 122 138 166 173 161 147 109 82 82 82 100 161 161 164 178 188 187 197 209 206 202 212 219 216 221 203 212 233 235 237 216 199 197 207 223 233 227 214 207 216 240 171 45 62 39 24 186 238 226 220 220 218 215 215 217 217 217 233 237 235 226 220 224 232 237 243 246 246 241 163 46 128 242 243 231 213 195 210 229 239 250 251 251 251 249 250 253 252 245 241 243 232 204 159 128 119 113 101 93 83 79 85 82 92 114 124 139 37 35 37 29 40 37 33 44 58 70 76 73 76 93 113 127 131 136 157 179 196 194 193 201 211 216 218 225 232 221 181 52 57 40 21 178 231 221 215 217 217 219 224 231 235 231 220 225 229 230 229 230 230 227 222 213 229 201 84 29 128 217 214 243 255 189 142 179 231 250 247 243 245 246 240 241 245 231 211 207 192 197 183 161 150 142 132 121 105 101 118 139 170 198 196 190 47 40 38 28 23 29 46 45 33 29 24 14 22 4 14 26 24 20 21 36 38 40 47 61 75 85 90 91 100 133 127 79 55 44 49 142 184 194 200 205 208 210 216 222 223 223 235 231 230 235 241 240 234 226 226 236 238 162 44 63 207 245 189 220 248 247 215 180 184 214 226 225 228 217 190 184 203 207 203 221 226 242 251 254 255 250 235 221 221 229 234 240 247 250 242 243 17 45 26 22 36 32 19 24 30 19 7 16 34 40 54 48 38 35 26 31 32 35 36 34 32 34 36 35 27 14 60 107 83 55 29 34 43 41 44 55 67 77 88 98 107 113 112 122 140 159 164 164 176 196 213 216 221 150 35 80 193 248 217 182 201 249 254 232 174 119 143 194 218 220 225 245 254 250 250 250 254 252 252 254 255 255 255 253 251 251 251 252 253 254 255 255
## 2534                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  133 96 77 43 29 26 18 104 114 48 60 40 44 52 61 68 71 78 85 92 84 87 80 61 49 53 46 31 28 30 21 21 23 18 16 17 17 21 25 29 30 34 39 61 55 48 56 57 63 58 57 82 57 53 69 57 49 43 46 45 31 32 31 30 46 76 58 29 45 34 29 38 49 29 30 81 59 42 58 38 149 209 206 213 206 212 208 208 208 208 208 205 205 207 208 210 113 93 70 31 18 12 58 149 129 72 62 53 51 61 69 78 89 93 100 110 107 97 79 63 57 67 51 34 47 42 38 41 24 30 28 30 23 20 27 31 32 40 42 63 72 69 66 54 74 85 72 56 79 80 69 69 60 55 52 49 44 36 38 55 51 43 77 71 25 38 49 35 39 20 17 65 83 30 45 40 109 220 206 210 206 208 208 208 208 208 208 207 208 208 207 207 82 78 47 36 34 30 100 169 104 75 53 58 64 73 82 93 109 117 111 110 117 112 81 66 73 79 66 68 57 47 56 62 46 55 44 38 30 37 48 45 46 60 55 60 80 86 79 103 87 75 91 82 101 106 92 83 70 64 55 35 30 37 42 55 65 49 38 70 79 31 43 28 51 46 14 27 100 51 34 43 86 209 208 204 207 206 208 208 208 208 208 207 208 208 207 206 101 83 55 48 39 55 154 147 118 91 55 60 76 88 96 103 112 115 116 117 121 114 86 81 88 78 69 97 86 70 69 64 73 77 67 61 47 47 57 60 57 64 66 69 77 93 105 118 105 84 91 89 115 119 118 106 82 80 76 61 28 41 63 42 39 66 44 27 53 47 49 22 46 53 32 17 69 71 44 31 69 176 212 201 206 208 208 208 208 208 207 205 206 207 208 209 137 98 53 52 39 95 171 162 88 64 64 60 79 90 103 114 122 117 122 115 116 118 102 105 109 84 75 102 129 119 102 94 84 73 86 76 76 73 64 89 89 49 64 67 69 88 106 102 103 105 105 119 124 99 110 126 113 95 76 74 94 36 34 42 59 72 51 61 36 35 33 28 30 25 33 28 49 81 49 62 42 104 202 208 206 210 208 207 208 207 208 207 206 208 209 211 97 90 96 77 42 150 172 172 87 63 65 63 88 98 107 115 124 131 123 115 115 119 110 120 105 102 96 126 147 130 123 127 117 77 85 90 104 104 99 90 83 85 74 70 76 100 91 95 97 119 143 149 129 139 133 136 127 109 94 69 80 62 68 36 50 56 52 57 57 50 50 61 38 20 33 28 27 46 48 44 42 74 187 210 209 206 207 207 208 208 208 206 206 208 209 211 128 102 100 75 98 180 167 165 94 72 61 85 97 106 112 119 129 135 114 111 115 124 121 113 111 103 112 129 154 152 131 119 135 121 135 124 100 98 95 121 111 103 103 114 95 104 86 107 119 135 141 148 148 140 145 144 142 127 99 68 52 58 88 52 61 51 61 62 80 74 68 71 62 42 38 28 43 27 45 41 43 53 180 206 209 207 208 207 208 208 208 207 207 208 209 210 118 92 100 43 113 169 158 155 92 52 73 84 97 108 118 127 134 136 115 114 120 123 132 109 114 97 138 145 137 143 141 147 131 132 134 130 121 107 112 128 132 124 110 136 131 130 112 128 123 139 147 147 154 145 155 151 155 145 105 64 53 73 49 64 58 78 73 73 81 60 46 64 64 54 29 38 43 32 57 55 50 58 168 208 206 209 209 207 207 208 207 208 208 208 209 209 89 103 82 74 157 169 166 111 58 54 77 88 97 104 123 139 138 128 117 117 129 130 126 120 118 117 146 149 142 142 140 136 143 150 146 144 138 131 121 119 133 133 143 135 130 136 119 137 144 151 152 154 155 159 147 149 150 149 105 80 45 53 77 69 75 92 61 81 74 69 78 57 73 73 47 54 45 17 52 68 61 49 115 210 208 210 205 207 207 208 207 208 208 208 209 209 92 133 74 58 150 160 186 105 51 64 65 82 98 116 122 131 136 129 124 133 134 128 130 134 130 135 136 147 141 144 132 149 160 158 159 160 156 149 145 146 154 143 144 139 145 135 134 145 156 150 158 163 158 156 147 153 157 150 123 77 74 46 63 84 100 106 111 57 58 61 73 66 56 64 58 58 43 53 39 39 47 23 101 200 206 205 207 209 208 208 208 208 208 208 209 209 135 115 47 89 174 161 157 65 47 60 72 86 99 118 127 129 134 132 126 137 133 125 135 137 129 139 141 145 148 156 155 156 156 156 156 161 165 160 164 160 157 154 152 146 150 147 149 142 141 156 167 166 159 159 161 153 157 145 118 95 89 56 64 82 76 102 95 96 53 57 60 61 49 63 66 77 78 63 70 47 63 33 64 163 209 208 208 208 208 208 209 208 208 208 209 209 143 83 47 124 174 173 129 33 48 55 75 94 101 107 127 129 133 141 138 138 133 128 132 138 129 143 149 140 150 146 148 157 161 165 164 166 172 176 183 182 164 153 165 145 140 146 149 147 147 159 161 159 156 156 159 153 167 151 121 101 74 80 90 101 90 85 42 71 86 49 63 61 55 60 63 75 75 63 85 66 56 41 24 140 218 204 206 209 208 209 209 208 208 208 209 209 90 79 66 134 151 163 96 46 52 51 70 94 100 102 117 123 133 142 142 143 141 140 134 140 137 143 147 143 147 147 154 162 168 173 172 172 177 187 188 189 181 165 181 171 160 156 148 153 163 165 164 164 164 159 153 153 169 148 135 116 89 92 63 80 120 83 77 63 61 74 49 58 100 62 81 78 94 87 80 99 67 86 39 94 198 209 208 208 208 208 209 208 208 208 209 209 115 71 70 139 174 166 75 33 40 51 63 82 95 110 116 124 133 140 145 146 143 137 136 144 144 153 148 153 151 159 166 163 170 171 174 176 182 197 202 198 197 178 175 169 170 167 155 163 175 180 167 161 163 159 154 158 161 143 123 102 85 90 110 98 139 123 104 49 53 49 81 49 66 79 78 72 75 91 95 67 54 62 50 55 166 216 209 208 209 208 209 208 208 208 209 209 105 73 65 155 163 126 87 36 46 60 61 76 90 112 119 125 129 138 144 145 144 142 144 146 147 157 152 156 156 153 154 154 165 171 171 177 188 199 206 205 205 190 176 171 177 175 170 170 170 182 168 163 162 158 149 154 161 148 107 92 101 80 100 138 116 115 94 65 66 79 56 66 62 95 84 85 83 86 67 70 67 66 72 61 175 212 206 207 208 209 209 209 209 209 208 208 90 81 57 154 138 57 63 34 44 65 72 91 101 114 125 128 133 142 146 147 145 145 148 149 150 154 154 152 155 149 153 157 161 173 172 181 196 203 200 205 204 190 180 176 173 176 179 172 162 168 168 166 155 156 156 158 165 143 125 104 104 78 132 95 108 90 98 110 66 63 69 64 64 79 96 106 86 73 54 78 77 41 55 52 198 210 205 206 208 209 210 210 210 210 210 209 95 56 91 145 104 58 47 36 39 59 68 82 100 109 116 120 128 135 142 147 147 145 148 151 152 156 155 152 151 154 162 164 169 176 182 188 195 201 206 204 199 193 188 184 184 185 179 175 176 176 174 165 152 161 163 155 162 124 131 118 115 98 131 89 85 132 99 129 77 61 68 64 66 91 88 80 93 81 79 61 77 63 67 39 132 214 206 208 209 210 210 210 209 209 209 209 85 57 87 126 90 60 52 27 27 45 69 86 91 105 114 119 126 135 142 147 144 141 145 150 150 154 153 150 152 159 165 161 165 176 184 188 189 190 194 191 187 188 190 192 195 191 181 179 186 180 166 160 155 160 160 146 148 120 135 127 122 116 100 120 109 146 147 106 91 74 53 74 87 76 103 80 69 67 80 68 84 68 65 37 136 204 205 209 209 209 209 210 209 210 210 209 74 40 72 101 89 75 43 32 33 38 61 78 86 100 113 121 129 135 139 144 143 139 145 148 143 147 148 149 157 162 164 157 162 178 183 190 195 191 191 192 186 180 179 187 195 194 193 185 180 178 166 161 160 164 162 152 142 145 132 149 123 110 99 120 101 122 138 125 104 103 72 78 99 86 102 97 73 89 103 97 68 68 54 57 119 214 209 208 209 209 208 209 210 209 210 209 48 39 61 87 76 55 33 36 41 39 58 72 86 94 102 111 124 130 136 140 139 140 146 148 140 141 145 149 149 159 164 164 172 181 184 190 193 185 182 188 191 185 187 194 196 190 189 188 181 172 169 162 156 164 169 167 143 167 141 130 128 117 102 115 146 145 146 162 128 94 106 79 86 117 109 95 76 109 77 75 113 84 63 47 115 213 207 210 210 208 207 209 210 208 208 208 36 38 72 60 59 45 37 33 35 42 64 77 91 96 105 114 121 126 138 138 139 147 145 146 144 135 139 150 151 158 163 165 167 173 181 184 187 183 185 182 184 183 196 182 187 195 188 188 180 170 165 159 154 165 161 154 133 146 139 143 113 104 117 128 127 118 121 153 134 89 104 95 100 106 96 98 66 123 97 118 119 60 82 60 146 217 212 206 207 208 214 211 207 214 215 211 18 29 47 60 54 41 37 32 42 50 66 80 86 94 111 121 119 129 135 131 135 140 139 141 141 134 136 146 149 154 161 165 167 174 172 173 186 186 178 179 181 192 195 193 197 200 190 186 177 169 159 156 155 162 162 160 127 124 135 139 109 66 137 118 119 109 140 125 136 124 91 94 82 120 98 114 78 72 116 112 107 53 71 79 158 207 208 214 208 210 205 211 211 210 191 177 29 23 53 57 38 42 36 39 40 57 77 88 89 92 107 122 122 131 138 133 125 129 135 137 141 139 144 151 144 147 154 160 160 170 176 176 180 180 186 196 190 201 193 197 198 197 189 183 175 174 161 158 157 158 164 158 139 115 122 132 110 116 110 102 128 103 136 155 111 113 106 94 108 102 129 109 82 68 103 101 135 92 45 41 176 196 216 201 207 216 207 212 194 188 185 191 44 44 58 71 70 43 34 38 34 56 79 93 101 97 105 119 127 126 130 133 127 132 140 138 137 134 138 145 147 147 156 167 167 179 196 196 190 190 200 201 199 199 188 194 196 191 188 180 171 170 161 157 152 156 155 147 148 116 121 113 120 131 77 130 124 117 98 120 127 109 89 108 99 87 112 74 97 82 77 60 100 108 55 51 165 206 212 226 208 199 187 195 193 197 210 215 77 51 99 77 60 45 39 35 47 61 75 99 110 110 111 119 129 129 119 129 140 137 138 137 132 131 137 144 143 145 161 178 180 186 194 196 199 200 199 190 209 201 199 201 200 193 187 178 168 164 163 160 153 165 150 153 145 110 118 102 111 98 84 97 93 123 97 78 126 78 91 102 86 108 112 72 58 80 85 63 90 94 71 63 135 218 169 148 187 190 198 209 211 223 223 221 117 56 97 99 76 75 48 42 51 65 86 106 118 114 119 123 136 133 122 126 137 134 137 138 131 132 137 141 144 152 164 174 180 185 187 195 195 186 189 194 195 196 198 197 189 189 186 181 172 169 165 159 157 157 164 149 128 112 92 117 90 112 119 87 89 80 90 66 83 113 85 70 81 91 101 115 69 62 68 98 72 87 72 57 90 82 68 60 157 218 218 224 224 224 222 222 120 104 76 71 61 75 39 44 51 63 84 104 116 131 131 124 130 139 134 133 134 133 133 130 130 135 138 138 141 149 164 173 174 180 194 199 189 196 196 193 191 194 195 194 188 187 185 180 172 172 167 162 160 155 164 151 138 116 103 95 94 99 93 81 83 98 67 72 76 97 76 57 100 122 81 96 93 55 43 75 91 85 69 66 37 41 47 55 116 219 230 225 222 223 223 223 89 129 101 72 57 22 31 38 51 59 79 102 111 140 142 138 140 137 132 134 128 126 131 127 127 134 141 145 147 154 170 181 178 188 185 191 198 202 200 198 190 189 189 191 193 191 184 179 174 173 172 167 160 157 148 148 123 92 114 95 104 95 85 90 68 102 46 54 90 82 85 59 81 114 107 85 95 72 66 80 87 94 96 37 24 38 34 38 75 200 231 226 223 224 224 224 77 80 85 81 39 20 23 31 43 52 74 99 107 122 130 137 132 130 130 137 131 129 136 133 126 130 141 148 147 155 165 173 173 181 185 189 189 194 196 197 192 192 195 194 188 187 178 175 175 170 172 166 158 157 160 148 143 99 89 61 63 77 104 104 85 99 69 55 62 73 83 36 79 96 132 102 104 83 79 57 62 76 93 57 31 25 25 37 53 164 223 227 224 225 225 224 38 46 76 27 17 13 21 21 23 34 58 81 112 115 128 129 127 125 130 137 135 128 137 138 129 129 135 137 142 152 158 167 179 174 184 194 189 189 199 205 202 201 198 195 188 188 176 174 178 171 173 166 159 164 171 163 158 119 77 63 76 58 81 96 63 69 96 55 70 99 99 58 55 89 100 114 102 78 94 94 84 80 67 61 48 31 26 31 37 131 213 226 224 225 225 225 26 35 26 18 15 12 10 19 16 26 49 66 95 116 124 118 123 130 132 137 139 133 131 138 126 128 134 136 142 151 154 154 173 176 176 191 197 196 193 199 190 193 195 186 184 180 172 177 176 174 172 168 165 166 163 166 152 143 120 75 63 61 44 109 82 82 80 85 55 68 89 83 55 100 95 107 98 60 71 87 66 85 70 47 45 39 32 27 27 94 208 225 225 225 226 223 34 46 35 24 13 16 18 13 15 17 27 32 64 95 106 99 101 117 137 149 138 137 130 135 127 127 127 134 135 141 148 150 160 170 170 169 170 183 182 187 182 186 193 188 184 179 175 180 180 172 167 166 166 160 168 169 163 163 145 81 68 82 49 84 93 107 97 95 87 76 77 64 63 81 79 80 90 76 56 63 73 102 51 85 51 41 34 29 26 57 173 224 228 224 229 222 48 79 70 53 47 39 25 33 34 30 21 23 33 59 112 144 100 98 90 101 110 134 136 120 123 113 110 135 146 145 142 146 141 157 156 148 147 150 146 147 159 175 180 186 178 179 176 168 181 174 171 171 168 164 167 169 169 169 165 138 100 61 59 88 86 114 55 87 70 81 78 43 54 54 88 82 80 70 38 61 63 112 71 123 56 48 33 28 30 31 124 223 229 222 228 224 130 126 104 81 86 83 54 40 50 50 40 41 20 16 42 97 133 58 47 39 45 60 57 71 60 64 113 111 119 123 117 117 121 102 94 97 117 121 121 114 130 155 159 165 171 175 171 160 175 175 176 175 171 168 165 162 168 170 174 166 121 59 55 66 67 39 91 76 86 72 67 23 55 62 58 85 109 105 49 51 43 81 89 170 151 40 49 30 33 20 93 211 225 220 224 223 88 90 76 47 43 48 60 59 56 56 65 88 49 75 41 24 13 18 51 34 28 22 32 43 58 139 74 65 60 57 64 58 67 75 83 73 72 71 81 87 102 120 129 138 161 161 161 161 167 168 173 171 166 162 168 160 169 171 174 177 167 134 69 39 44 61 64 61 128 99 93 38 47 57 48 80 103 85 67 55 50 57 69 174 181 41 62 36 35 21 63 183 227 224 221 221 3 70 24 2 10 0 3 6 10 17 18 39 37 50 90 116 84 75 61 49 59 63 75 69 64 156 42 52 40 27 44 41 52 83 111 103 85 76 70 76 95 94 100 104 134 154 143 158 164 162 166 173 166 164 171 160 167 170 170 172 163 163 109 45 52 43 76 96 116 98 79 53 35 70 63 55 72 84 92 68 59 49 75 161 228 121 47 42 34 27 35 154 227 228 222 223 42 134 60 9 12 14 1 5 14 3 6 5 3 9 21 67 170 122 58 51 44 66 71 53 46 62 52 65 76 73 96 109 134 151 160 171 165 146 135 123 124 125 131 114 101 101 108 128 158 158 150 163 163 167 173 164 165 167 169 161 160 158 79 41 54 53 66 94 126 143 85 92 50 101 75 48 61 67 86 74 72 60 76 173 242 115 47 54 41 31 27 124 213 224 219 225 54 160 69 30 13 3 29 31 10 7 14 12 14 9 16 5 62 99 57 40 53 78 96 51 40 62 74 79 81 69 64 63 56 63 60 58 64 75 76 81 95 135 141 134 115 124 103 104 104 138 146 158 159 165 161 164 152 156 162 160 142 84 40 80 98 94 91 90 92 66 83 74 58 77 65 73 79 109 116 100 86 75 87 157 228 165 77 51 43 38 25 79 197 226 221 222 65 180 119 50 16 10 35 41 11 19 8 13 11 6 20 20 9 66 79 38 50 86 101 68 51 62 81 58 53 44 36 42 28 17 17 16 15 15 17 27 42 67 66 89 114 131 139 157 147 95 121 149 159 142 160 155 149 155 152 122 65 50 76 130 132 116 86 64 81 69 45 65 86 62 69 84 99 126 121 104 95 103 92 144 228 214 79 74 49 39 28 56 167 226 224 220 68 192 105 94 34 73 18 8 19 6 10 15 11 17 13 11 16 57 98 64 33 64 87 77 53 71 102 77 80 59 39 36 25 24 19 13 15 20 20 17 13 11 23 48 72 90 113 115 145 178 101 95 148 141 140 140 148 131 97 67 77 99 116 169 132 113 76 67 85 74 28 41 61 45 55 70 104 138 138 107 110 107 117 216 206 212 92 67 55 41 29 43 123 216 225 223 92 175 95 106 98 78 28 21 28 17 13 9 17 21 12 19 9 16 75 38 45 66 85 89 65 92 123 94 82 58 37 25 29 29 26 25 26 35 19 14 11 18 43 17 8 28 58 95 106 115 150 85 91 138 130 132 125 80 54 80 85 104 165 174 153 137 88 92 93 68 33 37 45 68 75 56 102 132 145 105 128 122 159 214 203 223 177 48 74 41 40 34 87 202 224 227 135 164 105 119 92 39 26 25 26 28 26 25 26 17 13 8 7 24 36 14 60 72 96 98 63 107 117 109 96 64 52 70 71 35 42 32 29 62 37 8 13 18 16 23 45 25 18 30 57 100 124 104 50 90 125 108 77 76 77 77 90 160 172 161 155 124 114 99 109 60 38 39 40 80 132 93 90 133 145 125 137 109 209 214 210 216 190 77 68 56 36 27 68 173 220 227 153 175 119 129 101 29 32 32 37 39 41 40 31 26 17 12 8 23 41 24 70 70 92 121 72 79 109 115 107 98 101 102 88 43 44 39 51 81 71 24 16 14 8 59 110 66 41 33 27 44 82 101 43 62 71 69 77 89 68 80 137 164 157 165 147 115 142 106 105 50 85 46 50 110 157 119 85 146 134 126 133 189 215 209 205 210 235 107 39 59 40 41 44 133 221 217 155 134 124 111 68 33 32 35 35 36 35 35 31 25 20 15 9 22 31 20 60 82 101 103 41 75 103 97 121 124 111 84 86 55 47 44 54 57 77 80 50 57 100 119 129 106 91 97 82 53 60 93 48 40 52 83 76 66 78 135 152 156 160 175 140 148 114 111 112 84 87 64 86 140 187 127 116 153 128 128 152 208 212 209 214 205 224 116 27 52 38 38 35 88 187 195 148 136 105 59 48 35 35 28 29 30 32 32 29 26 21 18 13 45 29 32 87 136 181 177 86 39 111 113 143 128 110 98 85 58 47 44 57 75 88 83 100 136 140 132 139 137 128 129 126 112 84 93 68 36 58 76 44 82 134 153 153 168 159 150 156 133 107 107 106 78 77 103 123 141 163 125 121 159 139 130 192 211 204 211 213 210 209 218 55 30 42 35 32 34 118 160 144 129 56 42 40 42 35 32 32 31 32 33 33 30 25 21 12 59 24 55 115 150 183 200 146 67 84 112 142 149 134 128 107 66 60 58 48 54 79 102 127 137 146 152 142 140 138 128 125 120 106 113 72 38 48 49 40 149 157 156 165 161 152 157 140 111 119 82 101 94 99 131 146 144 135 130 148 156 129 143 189 207 213 211 215 215 209 219 178 40 38 74 119 107 94 176 121 67 38 37 47 51 57 52 49 44 42 38 37 39 29 17 27 24 14 76 120 145 189 188 170 108 59 139 122 164 153 145 136 96 72 68 66 66 77 76 86 117 128 125 145 145 130 122 123 118 114 118 76 33 44 38 100 155 157 153 158 159 152 154 126 106 104 74 82 107 146 158 152 141 128 143 160 155 104 150 226 205 206 212 210 209 211 211 215 175 154 184 220 229 221 222 92 49 33 36 40 46 53 60 59 53 54 51 47 39 39 16 38 25 34 73 115 154 197 184 172 140 92 108 160 136 160 150 135 133 107 91 84 90 113 104 108 128 130 137 144 148 136 128 117 101 106 120 108 51 47 86 172 164 149 160 162 154 152 145 138 132 90 80 76 108 166 147 155 186 174 181 171 119 129 211 204 206 208 208 210 212 210 210 208 216 221 212 213 217 234 201 53 35 32 31 35 36 43 52 53 53 56 53 51 42 24 34 33 14 61 88 118 166 203 170 172 169 119 87 156 139 136 162 130 109 128 135 110 96 98 104 125 134 138 152 160 160 156 139 114 113 123 114 122 86 79 134 155 157 161 173 161 151 146 150 146 122 108 80 95 128 155 141 163 192 178 173 134 116 169 210 210 209 208 209 210 210 210 210 210 208 205 224 154 89 113 49 59 35 27 26 27 28 33 39 42 44 50 48 46 36 28 42 22 13 58 97 132 173 205 178 180 168 140 105 99 151 128 130 131 119 119 119 117 124 119 115 124 127 129 136 143 150 151 135 125 121 111 115 141 51 55 154 155 163 166 162 159 150 145 149 146 124 112 96 107 136 170 168 183 188 165 143 136 132 174 213 209 208 209 209 209 210 208 208 208 207 211 192 61 46 0 18 74 40 28 30 24 21 23 29 31 33 39 36 28 28 33 18 13 24 66 97 141 175 188 173 168 167 172 131 110 159 154 122 120 122 113 106 112 115 112 117 124 124 133 143 142 154 156 139 131 127 122 126 136 58 54 161 156 155 160 159 156 147 138 144 149 134 119 134 117 127 178 188 185 183 158 151 132 140 187 213 201 210 209 209 209 209 208 208 209 205 226 90 32 69 26 14 86 89 37 26 28 21 22 27 30 29 29 28 28 36 22 18 20 32 83 107 143 182 189 170 169 170 169 166 113 103 178 161 115 117 110 99 101 103 105 118 120 122 129 138 151 165 176 157 143 134 127 129 141 72 66 157 157 166 162 156 150 141 135 144 150 142 127 117 108 140 184 186 176 164 139 150 127 175 207 205 211 209 209 209 209 209 209 208 208 207 220 147 46 21 15 18 37 76 100 79 30 35 26 26 28 31 23 26 28 24 19 25 22 21 66 116 153 197 194 172 170 173 169 169 163 116 104 180 146 111 98 102 88 96 100 101 97 117 133 133 145 159 183 175 154 141 133 130 142 65 98 164 162 162 147 152 146 143 137 137 147 147 117 106 110 140 167 153 166 148 140 126 133 191 211 206 208 208 208 208 208 208 210 208 206 208 210 228 66 28 25 14 27 11 26 35 50 54 32 26 24 30 28 23 27 33 22 17 20 23 68 122 159 192 181 169 167 170 171 171 184 182 149 116 134 143 95 79 79 84 87 87 93 98 104 109 134 140 165 176 151 139 133 130 146 50 135 163 165 159 148 150 140 138 131 134 138 124 107 95 133 163 148 167 160 131 140 94 150 183 203 209 208 208 208 208 208 208 211 209 208 206 228 90 21 75 25 15 32 31 36 31 32 26 31 29 31 31 38 44 50 45 23 10 15 29 82 131 155 194 180 168 172 174 169 169 183 161 146 136 130 96 168 107 75 75 75 73 73 76 90 97 113 122 135 152 144 133 128 139 121 66 157 166 166 150 151 142 135 132 122 126 128 120 101 94 149 173 174 162 144 146 88 116 191 182 219 203 211 209 208 208 208 208 208 208 208 207 215 178 40 43 45 19 31 47 60 70 84 87 81 78 80 68 66 64 54 38 18 16 16 34 89 136 155 203 182 165 171 170 166 165 170 176 147 112 131 112 136 130 146 92 76 65 63 77 79 80 94 110 121 143 154 129 120 134 77 105 166 162 165 154 158 140 134 130 115 115 113 108 94 112 153 165 161 150 133 80 88 170 194 197 210 206 207 209 208 208 208 208 207 207 208 209 207 222 186 91 33 25 28 40 47 67 82 92 90 85 86 79 70 51 40 24 20 33 26 43 91 135 159 203 174 159 160 162 165 175 168 174 164 138 83 130 146 135 146 139 123 117 91 83 67 66 66 85 103 133 150 127 124 74 79 152 165 159 160 159 149 133 127 124 111 115 112 98 90 126 139 134 127 100 51 30 134 200 198 201 210 209 208 208 208 208 208 208 208 209 210 211 212 204 213 201 154 169 23 24 36 49 61 68 74 82 70 63 64 45 27 17 32 36 29 50 96 135 168 208 170 155 157 160 163 167 155 165 172 169 118 76 120 160 170 147 129 107 136 92 68 110 111 112 92 80 94 96 65 85 160 159 153 149 153 158 145 134 132 118 108 115 110 99 103 91 74 73 44 20 24 57 181 183 199 205 210 208 208 208 208 208 210 210 209 208 209 209 209 209 206 208 222 219 31 22 23 33 44 50 57 66 60 54 49 33 19 17 42 39 39 54 102 133 175 206 169 163 157 156 161 168 168 185 169 163 147 49 55 107 147 184 193 149 139 113 106 96 89 86 69 67 84 101 122 163 163 160 161 149 148 150 145 132 119 117 119 110 102 99 78 52 29 10 17 23 17 74 177 204 203 203 208 209 207 208 208 208 209 209 209 208 208 208 209 210 211 209 205 205 33 14 18 23 32 39 47 53 53 49 42 26 11 11 23 43 42 52 103 124 171 202 166 156 151 156 165 175 174 174 160 153 121 56 24 49 95 132 158 178 200 183 168 151 151 154 144 141 148 155 141 150 154 158 156 149 143 145 148 134 113 119 115 97 95 88 67 55 44 19 20 26 26 77 156 195 210 209 206 206 206 206 207 208 209 208 209 209 209 208 209 207 208 210 211 211 91 15 16 16 22 31 36 37 41 38 30 20 13 9 3 37 41 58 110 124 168 199 172 152 154 164 170 167 160 159 143 119 99 117 78 29 42 93 132 144 166 179 179 186 182 191 184 166 150 148 155 149 154 152 146 144 138 139 141 130 117 118 105 90 92 89 80 69 52 27 26 23 34 85 150 163 185 207 213 207 204 206 206 207 208 208 208 209 209 208 208 208 208 205 205 205 110 36 15 14 13 22 25 24 28 27 18 14 13 9 10 12 33 63 107 124 155 175 167 147 147 166 160 136 104 76 99 121 141 144 113 63 28 43 86 135 144 156 172 170 173 172 168 158 154 146 142 148 153 144 147 139 138 134 128 121 117 109 105 101 88 89 82 77 61 38 28 25 68 148 157 173 160 166 192 212 210 204 207 207 207 207 208 208 208 209 208 212 208 211 219 225 117 69 17 14 17 13 20 22 20 16 11 12 12 9 8 8 22 48 84 115 135 149 151 146 147 120 79 65 57 92 126 146 153 137 123 103 43 23 47 83 117 140 155 159 152 157 156 160 155 140 144 144 146 144 139 136 135 131 123 117 116 106 100 91 84 93 91 77 73 52 46 72 117 176 176 160 159 161 164 183 205 208 204 209 209 207 209 206 211 204 210 206 227 155 88 95 99 93 34 15 6 11 18 15 14 12 13 14 10 8 4 8 14 25 45 81 117 135 135 117 84 50 58 97 116 144 148 146 147 138 134 127 81 39 25 43 79 113 133 139 143 145 146 149 140 139 141 135 136 135 130 131 123 122 119 115 113 102 93 84 86 100 93 78 84 81 113 133 145 154 174 174 163 172 169 147 162 200 209 204 191 194 194 207 208 215 207 195 178 70 52 14 93 103 60 17 12 14 19 18 12 10 10 9 6 8 7 6 8 11 15 34 63 77 76 58 38 57 98 116 127 142 149 147 142 139 134 130 112 70 25 20 40 75 103 120 129 130 135 140 134 138 138 135 132 126 125 125 120 119 116 114 111 98 91 88 97 100 94 86 101 132 157 176 169 151 164 184 175 169 158 151 170 161 184 207 183 115 78 96 139 157 123 190 151 7 56 14 103 95 96 22 19 17 25 24 13 10 11 11 7 10 10 9 9 14 15 14 17 26 32 46 59 79 106 111 122 135 140 140 133 141 138 125 110 87 51 27 23 42 71 100 111 113 118 121 127 126 128 132 126 116 117 121 120 120 114 108 103 90 87 87 100 96 87 89 128 167 163 173 181 167 157 178 189 162 159 171 163 163 151 174 157 80 50 27 34 73 53 83 106 45 29 5 94 80 100 59 14 21 34 32 13 10 12 9 7 10 11 11 14 12 14 18 31 47 58 73 70 78 108 114 118 127 128 137 142 140 135 134 120 104 75 46 25 20 38 69 84 103 111 108 123 124 120 123 114 110 117 118 115 117 115 108 99 91 88 91 101 85 89 93 150 183 180 177 185 183 176 170 180 186 167 161 162 146 155 161 164 126 70 32 78 180 117 45 98 56 36 32 70 87 104 112 26 24 32 44 28 15 8 8 7 8 9 11 14 16 17 22 36 59 68 84 90 87 107 121 123 128 126 127 131 128 132 134 120 100 91 62 43 29 24 39 59 89 102 111 122 126 121 120 116 114 117 116 114 116 114 107 97 85 86 94 96 81 99 93 154 195 189 186 176 182 189 189 177 175 183 162 150 160 159 163 153 143 138 142 160 182 178 208 139 0 64 58 67 108 120 106 47 18 37 41 42 24 9 11 7 7 7 10 12 12 21 29 37 60 77 90 98 100 110 119 123 120 122 123 123 118 128 120 102 98 78 47 54 41 26 19 28 63 89 112 121 123 123 122 120 116 116 116 116 114 105 101 96 83 91 100 96 92 90 107 153 194 202 197 185 184 192 192 189 172 175 180 157 156 161 151 117 147 161 168 173 151 138 153 129 60 6 12 75 91 103 96 91 17 26 36 40 30 15 10 8 9 9 10 11 12 15 17 25 44 68 85 90 91 86 82 92 90 93 96 98 91 71 65 68 65 38 31 39 33 34 28 13 33 64 89 124 134 134 132 125 122 118 115 114 111 105 95 88 89 98 102 96 94 95 139 181 186 200 198 198 186 185 193 195 189 173 174 167 157 153 123 151 153 151 145 147 117 93 117 98 47 79 58 50 56 58 49 67 50 17 29 36 31 17 11 10 12 14 15 13 15 17 17 19 26 38 50 69 76 66 54 50 51 40 32 36 50 60 55 38 58 83 91 79 72 57 45 31 30 55 73 108 132 136 133 124 119 117 106 102 109 102 87 86 94 101 99 97 97 91 138 179 197 195 204 200 192 184 188 193 194 184 166 174 167 147 143 146 143 139 128 137 170 180 211 124 22 36 30 55 64 54 35 59 87 30 20 28 25 17 13 9 13 16 18 17 20 29 38 38 37 28 25 39 47 52 66 79 86 91 95 102 116 121 129 120 129 123 117 102 96 74 64 40 48 84 97 104 120 131 130 118 111 114 104 95 96 88 83 91 98 98 95 99 89 101 179 163 198 203 203 201 201 194 183 191 193 185 179 166 170 150 141 149 124 117 135 137 138 146 111 25 31 22 13 62 65 57 37 79 105 48 14 21 21 16 12 7 11 16 19 18 25 30 41 60 77 81 70 64 66 89 122 132 145 160 153 130 112 108 126 132 124 118 128 119 106 79 74 42 52 93 118 113 112 120 118 110 105 101 101 94 83 83 87 95 100 94 94 96 92 147 192 167 207 211 195 198 201 198 194 178 188 193 176 170 147 170 148 134 122 139 125 132 119 134 57 26 87 56 14 62 59 49 52 80 86 96 43 12 18 15 14 14 17 15 15 20 23 23 29 55 73 85 91 92 100 109 120 122 117 106 105 104 122 121 124 131 125 122 128 119 98 87 78 54 62 92 123 115 116 115 104 111 107 94 94 87 76 81 87 93 98 91 95 98 94 176 200 178 190 215 208 197 198 201 196 195 175 187 182 178 163 157 157 136 136 135 128 117 117 137 61 37 42 28 6 64 49 38 69 78 81 140 162 33 11 14 20 21 23 20 15 11 18 29 28 29 39 47 55 67 74 76 84 90 90 94 113 127 126 121 124 132 133 123 128 124 104 88 80 65 72 89 111 106 104 112 104 100 91 88 85 75 73 83 96 90 87 92 99 96 124 198 200 168 138 217 213 207 203 201 196 201 182 181 185 177 173 159 157 145 128 139 116 117 106 112 75 25 62 29 10 66 48 45 68 83 100 146 194 146 42 21 29 31 27 21 17 16 13 22 26 30 36 37 37 35 36 51 67 89 105 115 116 106 107 115 125 131 133 123 118 118 114 101 81 65 72 79 95 97 99 107 101 95 91 81 74 70 74 85 89 87 89 93 96 97 170 193 207 169 136 218 210 211 203 196 200 193 194 187 169 178 171 170 140 144 143 120 123 121 51 99 87 28 31 18 13 63 41 55 63 73 128 172 161 175 111 43 51 51 37 25 21 17 11 13 19 29 32 27 30 32 39 52 57 76 82 86 97 102 105 119 130 126 122 119 115 112 115 109 85 60 69 72 86 88 93 94 90 84 81 66 66 74 80 90 90 92 98 92 86 127 213 195 207 171 152 219 209 213 210 202 193 196 200 183 180 166 180 165 159 125 136 136 133 107 66 102 104 35 8 13 10 54 47 50 78 110 145 177 154 161 135 98 67 53 44 35 27 22 17 21 22 27 29 28 33 37 48 52 42 59 71 83 103 111 112 126 134 128 127 125 124 116 100 97 88 65 69 71 84 85 90 85 84 77 63 64 75 78 76 85 94 85 90 91 88 184 202 205 196 189 171 217 204 210 214 203 197 188 198 190 186 169 168 179 162 149 127 141 125 117 107 95 103 96 64 48 49 55 42 60 94 122 165 159 158 148 117 104 91 54 44 42 36 34 29 33 31 34 43 47 44 49 72 72 71 74 91 101 111 121 123 125 128 131 136 139 130 114 113 101 93 75 71 74 76 78 77 73 81 65 55 64 76 80 84 85 88 92 93 91 144 220 203 200 195 188 174 211 213 205 205 205 204 191 186 191 185 182 167 169 170 154 142 120 123 113 117 111 95 94 83 52 115 53 44 75 110 142 173 152 147 141 105 108 111 64 46 45 44 40 38 40 44 52 65 73 68 70 90 86 90 94 109 116 119 123 128 133 138 139 138 137 136 124 125 112 98 88 85 79 69 68 68 75 58 49 61 70 81 86 88 87 82 88 95 107 211 203 198 200 193 179 153 206 213 212 201 202 206 203 188 182 191 178 176 168 163 159 147 130 108 119 110 119 102 80 46 38 40 46 54 87 120 156 158 141 149 132 107 117 68 72 51 47 41 53 45 47 57 67 80 93 93 90 103 102 104 111 122 130 133 136 134 136 141 144 143 134 142 128 118 112 95 81 78 71 65 71 67 52 44 55 66 73 86 92 88 83 84 80 88 176 224 201 201 197 202 190 153 195 218 216 204 199 210 199 196 193 183 188 179 170 158 157 153 138 113 99 107 96 109 81 15 52 65 45 74 108 137 164 154 136 140 124 90 120 51 74 46 47 47 48 58 56 69 76 83 100 108 101 112 114 122 129 130 131 136 141 145 141 138 139 136 141 138 109 119 112 89 82 67 58 56 62 54 46 55 50 69 80 84 91 85 81 83 84 119 228 200 211 209 199 202 205 162 186 224 215 212 205 197 204 198 191 182 178 184 174 163 142 148 139 130 99 91 92 95 88 61 13 32 57 95 124 154 158 126 137 140 112 92 108 75 156 68 48 49 54 57 62 77 80 78 97 106 101 113 122 130 135 135 138 140 139 144 139 141 145 141 127 118 106 103 93 94 76 56 54 53 52 49 53 53 49 66 79 82 90 84 85 82 117 230 217 210 199 208 198 196 215 158 190 214 213 215 207 192 193 202 203 189 165 178 177 169 156 128 138 131 115 89 84 89 79 72 38 7 79 108 128 151 143 121 135 121 99 108 98 64 233 92 33 48 51 51 58 80 79 85 89 101 98 105 112 127 133 135 138 143 141 134 137 134 130 131 124 111 96 90 83 70 49 47 53 56 52 58 56 48 55 55 70 81 90 80 81 122 229 227 211 204 207 202 199 201 211 161 190 217 208 212 211 208 194 190 200 190 187 165 171 174 164 146 122 132 113 103 82 93 71 34 11 27 103 116 138 153 132 125 136 119 87 111 78 65 221 197 63 36 37 45 70 54 83 85 80 79 93 102 100 114 126 131 130 133 129 127 125 118 117 116 108 104 88 73 65 50 43 47 54 57 54 60 58 51 50 54 66 82 87 86 129 241 223 207 206 202 210 198 199 202 198 151 189 222 212 207 212 206 207 193 190 201 182 178 161 169 163 149 129 115 123 107 94 81 88 51 15 14 120 121 150 154 130 129 134 112 86 110 71 145 220 231 141 16 38 38 54 67 69 76 78 66 73 84 94 108 119 130 125 119 115 131 122 110 111 95 84 76 57 49 50 50 53 56 60 61 58 64 63 54 50 53 61 73 73 135 230 224 204 209 198 216 199 202 207 211 210 154 187 218 222 209 206 209 202 195 188 189 189 179 169 160 165 150 139 115 111 107 103 86 69 98 64 45 125 137 158 141 125 128 117 85 92 103 62 183 226 215 231 107 15 34 32 53 58 72 72 65 56 61 76 91 99 112 110 108 107 103 102 95 90 81 66 49 38 39 46 53 55 59 60 60 60 65 71 63 54 52 57 63 121 236 231 204 215 203 207 212 195 200 208 213 220 167 178 221 219 213 205 200 205 204 193 181 182 176 176 159 142 154 137 127 101 98 102 95 80 66 88 88 128 151 153 133 122 118 107 85 93 101 64 177 227 218 230 233 97 23 28 39 50 52 58 44 41 48 56 66 86 87 83 88 85 75 80 73 64 55 41 33 40 47 47 51 57 60 60 58 57 59 69 66 64 64 73 131 220 231 206 205 211 201 213 196 204 206 209 213 226 175 182 213 216 216 208 191 202 200 187 186 163 181 175 166 145 135 142 126 122 92 88 95 93 89 73 97 136 154 144 125 118 110 98 88 97 83 77 215 221 215 224 226 240 195 67 17 13 32 39 37 29 33 53 54 51 53 58 57 65 53 47 44 40 28 34 38 39 44 49 58 58 69 63 64 64 70 78 81 88 124 180 234 222 201 201 209 197 218 203 207 200 217 211 204 219 169 193 210 209 212 210 206 195 200 201 185 177 163 169 163 164 132 136 136 126 106 88 98 89 100 95 78 143 159 141 116 116 108 97 85 97 68 115 229 219 223 227 226 227 241 217 150 161 93 24 32 31 14 23 33 29 31 33 32 25 31 26 29 28 30 33 39 46 48 52 57 59 67 52 69 77 70 83 124 178 197 224 215 202 197 201 204 207 212 201 204 211 213 198 194 198 168 196 221 206 209 215 208 204 194 192 194 185 169 159 166 160 150 133 136 124 115 103 92 100 91 95 81 150 149 120 116 113 100 83 91 83 52 138 242 221 231 227 224 231 227 231 232 241 224 204 180 126 82 24 25 46 23 25 22 22 22 25 28 24 27 38 43 46 49 50 49 67 69 73 77 76 116 177 215 209 218 207 201 200 201 201 209 218 199 214 200 215 202 184 202 179 166 204 212 214 206 208 207 202 198 191 189 182 177 163 155 160 154 138 121 124 118 112 87 84 111 87 114 151 127 102 109 106 96 83 87 74 43 131 236 216 231 225 224 233 233 226 223 220 230 230 211 153 148 101 52 68 46 20 19 22 25 32 30 25 31 39 42 44 50 56 67 59 67 70 127 170 224 215 215 209 201 199 201 200 196 203 210 206 208 205 212 212 201 199 204 171 163 206 221 211 204 207 212 203 198 190 178 185 181 175 152 153 156 149 123 109 119 115 107 82 81 110 84 143 112 98 109 103 93 92 78 64 48 145 235 215 229 224 228 233 233 231 226 237 237 234 234 163 142 145 105 53 37 31 31 24 30 34 26 28 36 39 46 52 52 57 63 87 87 169 229 234 206 202 199 196 190 198 201 201 202 210 209 196 212 201 219 206 200 210 200 170 179 197 212 212 214 209 189 197 201 193 182 167 175 174 172 132 140 144 144 117 102 116 113 111 87 91 84 140 97 103 114 103 83 92 79 51 46 171 230 221 231 227 234 229 230 241 226 243 224 225 216 141 148 172 162 146 91 30 22 22 29 37 36 41 43 46 61 70 76 108 137 172 225 225 203 183 189 187 196 196 197 204 197 200 210 201 201 207 196 216 210 207 207 208 196 158 188 215 200 205 208 206 200 190 196 199 188 170 166 178 167 157 134 147 137 132 118 111 118 113 103 85 106 126 107 109 114 98 80 92 73 41 45 176 226 220 231 220 234 230 233 235 229 233 212 214 194 175 168 187 157 116 152 96 123 102 97 37 32 70 92 107 134 172 180 196 216 210 194 182 182 188 194 195 190 196 206 198 199 204 198 197 200 203 206 213 216 208 211 208 183 163 185 216 214 203 213 214 203 197 188 190 191 179 169 171 169 160 150 135 137 130 133 108 109 113 81 91 86 116 106 121 116 96 91 92 65 41 49 173 229 214 226 221 234 232 232 232 223 226 219 201 195 179 131 89 34 3 51 144 176 167 185 144 138 167 192 201 201 202 203 200 191 173 168 184 197 191 184 195 202 206 197 197 208 196 197 203 192 199 213 217 212 209 214 194 174 171 191 208 200 215 212 206 206 195 188 184 188 187 175 158 165 157 159 132 123 140 127 119 100 102 76 81 78
##       .src    ImageId      .rnorm .pos Image.pxl.1.dgt.1
## 1908 Train Train#1908 -0.35240028 1908                 5
## 1862 Train Train#1862  0.60621884 1862                 1
## 6493 Train Train#6493  0.79442596 6493                 1
## 4264 Train Train#4264  0.04966901 4264                 4
## 2534 Train Train#2534  0.94331596 2534                 1
##      left_eye_center_x.All.X..rcv.glmnet
## 1908                            66.27193
## 1862                            66.18836
## 6493                            66.57747
## 4264                            66.44447
## 2534                            66.21912
##      left_eye_center_x.All.X..rcv.glmnet.err
## 1908                                43.50859
## 1862                                30.02270
## 6493                                29.62811
## 4264                                28.24481
## 2534                                26.76403
##      left_eye_center_x.All.X..rcv.glmnet.err.abs
## 1908                                    43.50859
## 1862                                    30.02270
## 6493                                    29.62811
## 4264                                    28.24481
## 2534                                    26.76403
##      left_eye_center_x.All.X..rcv.glmnet.is.acc     .label
## 1908                                      FALSE Train#1908
## 1862                                      FALSE Train#1862
## 6493                                      FALSE Train#6493
## 4264                                      FALSE Train#4264
## 2534                                      FALSE Train#2534
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
##   Image.pxl.1.dgt.1 .n.OOB .n.Fit .n.Tst .freqRatio.Fit .freqRatio.OOB
## 0                 0     18     26     15     0.00514444     0.00906801
## 6                 6    109    263     97     0.05203799     0.05491184
## 3                 3    153    323    137     0.06390977     0.07707809
## 4                 4    132    329    117     0.06509695     0.06649874
## 1                 1    744   2000    674     0.39572616     0.37481108
## 9                 9     89    233     79     0.04610210     0.04483627
## 8                 8     80    268     70     0.05302731     0.04030227
## 5                 5    129    263    115     0.05203799     0.06498741
## 2                 2    434   1078    393     0.21329640     0.21863980
## 7                 7     97    271     86     0.05362089     0.04886650
##   .freqRatio.Tst err.abs.fit.sum err.abs.fit.mean .n.fit err.abs.OOB.sum
## 0    0.008412787        78.52908         3.020349     26        57.77299
## 6    0.054402692       540.63367         2.055641    263       256.84422
## 3    0.076836792       680.84293         2.107873    323       356.53763
## 4    0.065619742       624.38775         1.897835    329       306.05286
## 1    0.378014582      4125.29454         2.062647   2000      1685.02074
## 9    0.044307347       519.57464         2.229934    233       195.59648
## 8    0.039259675       601.16807         2.243164    268       173.24281
## 5    0.064498037       466.26785         1.772882    263       272.12393
## 2    0.220415031      2205.95815         2.046343   1078       884.47657
## 7    0.048233315       590.18984         2.177822    271       173.73053
##   err.abs.OOB.mean
## 0         3.209611
## 6         2.356369
## 3         2.330311
## 4         2.318582
## 1         2.264813
## 9         2.197713
## 8         2.165535
## 5         2.109488
## 2         2.037964
## 7         1.791036
##           .n.OOB           .n.Fit           .n.Tst   .freqRatio.Fit 
##       1985.00000       5054.00000       1783.00000          1.00000 
##   .freqRatio.OOB   .freqRatio.Tst  err.abs.fit.sum err.abs.fit.mean 
##          1.00000          1.00000      10432.84651         21.61449 
##           .n.fit  err.abs.OOB.sum err.abs.OOB.mean 
##       5054.00000       4361.39876         22.78142
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
## 1 fit.models_2_bgn          1          0    teardown 113.706  NA      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc=FALSE)
```

```
##         label step_major step_minor label_minor     bgn     end elapsed
## 17 fit.models          8          2           2 105.616 113.717   8.101
## 18 fit.models          8          3           3 113.717      NA      NA
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
## 18        fit.models          8          3           3 113.717 118.215
## 19 fit.data.training          9          0           0 118.216      NA
##    elapsed
## 18   4.498
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
## 1 Final##rcv#glmnet .pos,.rnorm              20                      2.362
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit max.Adj.R.sq.fit
## 1                 0.007  0.003445728     3.441975      0.003162455
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.002829642      0.1253077        0.001332354
```

```r
rm(ret_lst)
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=FALSE)
```

```
##                label step_major step_minor label_minor     bgn     end
## 19 fit.data.training          9          0           0 118.216 122.384
## 20 fit.data.training          9          1           1 122.385      NA
##    elapsed
## 19   4.168
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

#stop("here"); glb2Sav(); all.equal(glb_featsimp_df, sav_featsimp_df)
glb_featsimp_df <- myget_feats_importance(mdl=glb_fin_mdl,
                                          featsimp_df=glb_featsimp_df)
#glb_featsimp_df[, paste0(glb_fin_mdl_id, ".imp")] <- glb_featsimp_df$imp
print(glb_featsimp_df)
```

```
##        All.X..rcv.glmnet.imp Final..rcv.glmnet.imp imp
## .pos                     100                   100 100
## .rnorm                     0                     0   0
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
##      mouth_center_bottom_lip_y
## 1908                  76.62886
## 2788                  72.55607
## 1862                  82.33012
## 6493                  47.21769
## 6406                  54.71770
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          Image
## 1908 5 10 20 29 39 48 61 73 94 112 120 123 127 124 122 123 122 123 123 120 123 126 124 125 127 125 128 132 134 134 138 141 144 147 147 144 151 154 152 155 162 164 167 175 180 184 187 190 192 192 192 192 194 196 197 197 198 200 201 204 205 208 213 217 213 215 218 220 217 217 218 220 216 217 193 169 152 178 169 141 156 160 160 148 129 144 136 157 216 244 246 243 244 243 243 243 12 16 21 35 45 51 63 82 104 118 123 124 128 126 121 125 126 123 128 125 122 125 124 127 128 125 128 133 138 139 143 144 145 149 152 153 155 159 159 160 164 169 176 182 181 182 189 191 191 194 192 193 195 197 198 195 197 198 200 207 209 212 216 213 215 221 219 216 218 220 222 223 217 221 207 181 159 162 191 167 134 118 149 164 159 149 138 130 203 250 243 245 244 243 243 243 17 17 27 42 48 55 67 86 107 118 123 126 127 125 122 123 124 123 127 126 126 125 125 128 127 127 131 135 138 140 143 143 144 151 153 158 160 163 163 161 165 173 181 184 180 179 184 190 194 194 193 195 195 194 195 192 199 201 196 205 210 215 220 217 218 213 215 220 223 223 222 222 220 225 216 199 183 168 187 197 149 131 111 124 107 136 146 120 168 247 245 245 244 243 243 243 12 27 36 41 52 62 74 94 112 122 127 130 125 121 124 123 122 123 125 125 126 124 123 124 124 126 129 130 132 140 144 148 151 154 154 160 167 167 162 164 174 179 182 182 180 178 181 190 198 195 192 191 193 196 196 192 197 196 193 202 214 217 217 223 222 213 219 225 225 222 224 222 218 223 221 213 203 186 182 199 180 155 127 118 100 63 99 124 158 234 248 243 244 244 244 243 22 34 38 55 63 68 79 98 111 121 128 132 124 120 124 126 123 123 126 124 123 125 128 125 128 131 130 130 133 140 142 144 145 151 154 159 166 170 165 166 176 179 185 181 178 176 184 193 197 195 191 189 192 193 192 196 196 197 201 204 219 220 219 222 222 224 221 224 225 224 225 224 220 223 223 212 209 197 185 184 184 167 144 98 89 78 49 89 156 229 247 245 245 245 245 244 25 32 52 56 66 73 84 100 112 122 128 129 126 124 125 127 126 122 122 124 121 123 126 127 129 130 129 130 133 133 142 145 149 151 155 160 165 168 166 168 173 175 187 184 180 180 184 190 190 188 190 191 196 198 198 202 199 201 203 204 214 224 228 228 228 228 224 224 224 222 224 222 220 220 222 213 200 202 192 190 191 182 153 122 75 66 54 51 132 232 248 245 245 245 245 244 33 39 55 55 61 75 88 98 115 125 129 127 126 126 124 125 126 122 121 123 120 123 123 125 126 129 131 131 131 133 140 140 149 151 152 157 164 167 166 164 169 172 182 182 183 177 170 178 185 188 186 192 199 199 200 198 199 201 201 202 209 219 223 228 227 224 227 228 223 222 225 223 220 218 222 219 211 200 194 187 187 190 169 140 114 73 52 47 138 242 246 246 245 245 245 244 46 56 55 57 66 71 85 99 114 124 128 126 126 126 122 125 125 122 122 122 120 125 125 127 125 127 130 128 127 135 139 139 152 156 153 159 167 170 169 167 174 179 183 184 185 179 177 180 184 190 191 190 197 202 200 200 196 200 200 203 209 213 220 223 223 223 220 221 225 225 224 223 221 216 220 220 221 209 195 197 181 173 176 162 129 100 72 44 144 249 246 247 245 245 245 244 50 53 53 59 66 73 88 107 110 118 126 124 122 124 125 127 123 122 122 122 122 123 124 124 127 128 123 126 128 134 135 138 147 152 152 159 169 171 170 170 175 182 184 189 188 181 179 184 180 186 190 193 197 198 196 195 201 205 201 202 209 213 221 226 224 224 224 221 225 223 222 224 223 218 219 218 220 207 200 196 186 168 157 146 152 105 85 74 180 253 243 246 245 244 244 245 45 54 60 59 73 83 93 103 106 117 121 123 123 123 122 122 121 122 121 121 122 122 121 124 124 126 125 129 132 132 135 140 147 152 156 159 169 170 171 172 176 177 180 189 189 182 176 179 178 180 183 191 196 196 198 194 196 199 204 202 208 213 212 216 220 224 226 222 225 224 222 223 220 218 220 221 220 220 205 193 187 180 158 122 125 125 70 101 216 247 246 245 245 244 244 245 56 55 56 75 81 85 89 96 108 119 122 123 121 119 119 124 122 120 121 118 119 123 125 119 116 124 126 128 133 136 140 141 147 152 153 164 166 168 171 170 173 174 178 187 189 184 178 178 177 179 182 188 191 191 196 193 197 202 203 202 206 213 213 215 221 221 226 224 224 224 224 225 218 215 218 222 224 225 214 196 179 173 167 136 92 95 99 139 243 246 246 244 245 244 244 245 53 57 66 78 80 81 89 104 113 118 122 118 118 120 123 122 120 121 124 123 119 118 121 119 116 124 125 125 129 131 136 141 145 148 155 161 162 169 169 165 168 174 178 185 187 182 179 179 174 183 185 186 186 186 188 188 192 194 196 199 196 199 209 214 219 219 223 223 221 220 221 223 216 215 219 223 220 220 211 204 186 167 153 134 111 75 76 147 245 246 245 245 245 244 244 245 61 71 72 73 80 86 96 110 117 119 121 120 123 123 122 119 123 121 122 120 118 120 119 119 121 120 125 128 126 132 134 137 145 151 156 161 162 166 172 171 168 166 173 183 185 181 176 177 173 176 179 183 183 182 184 188 195 194 195 200 199 200 208 213 213 215 219 220 218 223 223 219 216 215 217 222 221 218 211 202 178 164 148 124 117 86 54 152 250 246 246 245 245 245 245 245 66 72 69 75 86 93 101 109 114 119 123 123 121 120 120 119 122 119 118 119 119 122 122 121 121 119 122 125 124 129 127 128 138 146 147 153 159 159 165 165 162 163 170 176 178 175 170 172 172 174 185 188 194 190 191 193 193 200 200 202 198 202 213 218 218 218 218 216 217 221 219 215 215 218 221 222 221 221 212 199 185 159 138 119 107 89 59 156 252 245 246 245 245 245 245 245 70 73 76 83 82 88 100 114 119 122 122 119 117 118 117 118 118 116 119 123 117 114 119 119 117 117 118 121 120 120 127 129 134 141 144 151 162 166 168 169 169 171 173 177 178 179 178 177 177 174 187 194 186 179 191 198 196 198 202 203 196 201 211 216 217 216 215 214 215 220 218 217 216 218 221 220 222 224 215 199 185 165 139 120 101 83 70 176 253 245 246 245 245 245 245 245 70 74 77 73 77 90 104 114 119 122 123 121 123 121 117 116 115 117 120 119 116 115 117 118 120 123 122 124 124 130 136 137 143 143 145 151 160 170 171 171 171 171 172 180 180 179 181 180 183 176 187 194 187 177 190 195 199 199 199 197 199 205 211 219 219 217 216 216 215 220 220 216 215 216 220 222 224 225 212 212 194 164 139 122 97 83 87 201 251 246 245 245 245 245 245 245 68 68 71 73 85 97 103 111 116 122 123 118 116 117 115 115 116 116 116 117 118 119 116 120 121 121 120 122 125 125 129 135 142 141 147 153 156 165 170 171 172 169 176 183 181 182 183 182 179 183 187 187 189 191 191 189 192 196 198 199 203 209 212 218 220 216 214 213 214 220 220 217 215 217 221 218 219 223 218 218 205 174 136 116 106 78 82 210 250 245 245 245 244 244 244 244 59 67 78 78 85 93 101 111 117 120 121 118 113 119 117 113 116 116 115 116 120 122 115 114 117 122 120 119 124 126 128 134 139 143 153 157 158 163 169 170 174 172 179 182 182 183 178 179 175 186 187 172 189 187 187 186 192 195 198 199 201 207 212 218 221 217 216 213 213 217 218 217 216 217 221 221 224 221 220 221 202 181 153 113 91 70 74 212 250 245 245 245 245 245 245 245 67 76 79 80 83 92 104 115 119 119 121 120 118 116 115 116 115 115 115 114 116 119 117 115 117 119 119 121 126 127 128 134 140 146 155 157 158 165 168 168 174 177 178 179 183 184 179 172 176 191 187 160 181 189 184 182 190 192 198 202 204 214 215 218 221 217 219 221 219 220 219 216 214 216 220 221 219 222 221 216 207 184 154 129 94 60 98 221 249 247 246 246 245 245 245 245 85 83 76 80 85 95 108 114 116 119 122 123 121 115 115 116 116 119 120 119 119 118 121 119 120 116 118 120 125 125 129 137 142 147 153 154 162 166 170 172 177 179 179 181 183 183 184 173 175 182 183 185 189 189 189 187 192 195 202 214 211 217 218 223 224 223 225 224 221 223 222 219 216 217 219 220 214 220 221 217 217 194 149 123 119 70 95 225 249 247 246 246 245 245 245 245 80 76 82 84 87 93 104 112 116 120 121 120 121 120 118 115 115 115 121 121 117 117 121 119 120 121 121 122 125 126 129 135 139 148 152 151 163 168 175 175 177 181 183 187 188 183 180 175 169 175 179 182 189 189 191 192 191 193 201 212 212 219 220 222 227 228 226 223 224 225 222 223 220 215 220 222 218 218 218 218 219 205 164 119 104 93 90 209 250 246 246 245 245 245 245 244 76 79 83 87 88 95 105 116 122 124 121 120 120 121 117 117 119 118 122 121 118 122 125 125 125 126 125 126 129 131 133 138 141 150 158 154 156 166 173 171 172 182 186 193 194 184 182 178 168 171 177 184 193 190 191 193 197 198 204 211 215 221 221 221 227 227 225 224 223 227 228 226 223 217 215 217 222 215 211 215 216 215 177 124 91 94 123 214 250 245 246 246 245 245 245 244 90 83 77 85 95 106 113 117 117 118 120 121 119 120 118 120 120 120 122 123 122 124 128 127 128 130 127 128 132 134 136 139 140 146 158 161 164 166 173 177 175 180 185 193 196 182 179 172 165 169 176 188 192 194 193 191 199 200 203 210 215 216 219 221 224 221 217 210 203 204 205 212 220 218 217 218 221 218 214 213 215 221 196 139 106 86 106 214 248 243 245 246 245 245 245 244 84 82 85 94 103 112 117 119 118 118 123 125 123 123 122 122 122 121 122 125 126 125 129 130 129 131 131 130 129 132 136 138 139 141 147 158 171 169 171 181 178 183 185 191 197 184 181 168 160 169 179 181 189 194 194 193 196 196 202 207 214 219 213 210 213 214 212 200 191 183 180 183 193 207 216 219 217 218 215 215 218 217 212 173 124 90 85 184 242 247 246 244 245 245 245 244 86 94 103 109 111 117 120 123 123 122 124 124 123 125 123 125 125 125 123 125 127 127 132 130 130 132 133 133 131 133 137 141 139 143 147 156 164 170 173 177 178 185 188 188 194 178 176 172 163 169 178 183 191 189 189 194 196 200 207 207 205 207 211 208 206 198 185 177 175 177 182 180 173 187 185 194 215 220 219 218 217 219 217 185 142 95 84 140 207 237 245 244 245 245 245 245 100 106 116 121 121 122 124 123 121 123 124 123 121 124 124 126 127 128 127 129 129 132 130 127 132 136 134 138 137 137 138 139 141 145 147 150 158 166 176 174 180 185 187 193 202 179 171 173 161 164 174 182 190 188 188 192 198 200 203 207 204 195 186 182 179 171 160 153 148 158 170 176 175 178 175 176 198 215 216 215 216 215 215 194 151 108 103 150 199 228 245 245 245 245 245 245 112 123 126 130 128 127 128 125 123 123 124 123 124 125 128 127 129 129 124 118 116 125 121 124 126 132 134 134 133 136 139 138 147 151 149 151 158 163 171 170 176 185 185 193 204 171 159 169 159 154 180 182 182 189 190 193 197 195 192 187 166 151 142 136 137 131 132 142 146 153 162 168 176 179 184 181 182 192 208 216 214 213 218 203 155 120 110 159 209 236 246 247 246 245 244 244 118 128 133 135 127 129 131 128 125 123 123 124 126 126 131 127 126 123 113 103 107 115 113 109 110 111 117 122 129 133 134 135 141 151 151 151 155 158 169 169 170 179 183 193 206 169 149 166 161 144 170 183 178 187 187 184 179 166 146 129 101 112 120 115 99 87 110 131 144 145 155 156 151 148 146 147 166 178 200 213 213 214 216 210 165 120 105 132 191 242 245 245 245 244 244 244 123 136 139 136 131 132 133 127 126 123 123 126 125 127 132 127 114 103 93 89 86 83 86 84 96 96 99 100 104 112 120 130 133 145 148 150 151 157 162 161 164 173 179 187 200 161 135 156 166 154 161 176 182 177 170 162 151 132 112 96 88 91 83 78 82 88 105 132 151 147 147 143 142 142 141 151 158 173 195 205 213 219 218 215 177 116 99 131 202 245 247 245 244 243 244 245 127 141 142 140 137 132 132 128 127 124 122 126 128 124 112 97 88 77 78 74 69 78 77 77 79 79 70 61 64 76 82 86 91 101 124 138 142 149 154 155 157 167 177 181 200 154 126 153 166 149 155 164 170 171 145 141 122 108 113 99 91 81 79 86 103 118 141 163 166 177 178 171 170 163 169 183 179 187 192 198 215 219 220 213 179 122 110 152 219 248 245 244 244 243 243 243 131 144 145 143 141 134 130 127 124 121 121 122 113 96 83 76 69 68 72 72 77 72 70 55 48 42 31 28 33 39 46 50 63 70 84 108 129 131 146 149 153 160 172 183 196 158 126 137 164 139 141 156 153 148 148 131 110 112 106 100 98 104 112 132 156 167 173 170 177 185 199 198 196 195 191 190 190 185 179 192 214 218 218 213 177 133 118 163 227 247 244 244 244 243 243 243 133 148 149 144 140 136 131 129 126 125 124 118 91 72 64 59 53 46 48 43 47 49 55 47 34 25 26 27 24 24 32 34 38 47 61 68 97 105 115 133 142 147 163 179 186 150 128 135 154 152 142 152 152 146 138 133 126 122 127 127 129 135 141 154 165 170 175 176 181 181 190 194 192 197 198 187 187 184 185 204 218 217 219 216 182 128 120 174 231 246 244 244 244 243 243 243 133 146 149 145 141 137 129 129 124 122 114 95 76 67 66 59 48 46 43 38 49 66 69 67 57 49 42 40 38 34 29 32 33 32 45 54 68 85 102 116 127 141 155 175 179 138 127 142 146 152 141 143 152 149 142 141 143 146 142 129 132 138 139 153 158 165 168 170 172 176 181 186 191 196 195 195 188 189 200 210 217 219 218 216 199 132 124 177 228 246 245 244 244 243 243 243 129 143 149 145 142 138 131 127 116 116 109 95 90 84 69 60 63 62 62 76 88 89 93 103 108 104 102 94 78 66 50 45 48 50 49 60 70 75 85 103 116 129 142 163 162 134 139 147 149 153 146 141 150 144 148 151 145 131 118 112 107 102 99 100 99 102 109 117 120 124 133 153 172 183 193 191 193 198 201 214 220 220 219 217 200 140 143 182 228 246 245 244 243 243 243 243 122 140 150 146 142 136 134 125 118 121 119 103 92 89 87 84 80 76 80 101 112 107 108 115 116 118 123 123 118 107 93 76 62 57 63 67 71 74 76 86 97 114 127 140 145 138 139 156 159 167 168 158 157 157 144 124 113 108 95 74 51 29 26 32 32 34 49 52 47 71 80 85 126 175 188 191 197 203 207 216 221 222 218 216 204 147 167 197 233 245 243 245 243 243 243 243 122 139 149 146 144 137 133 118 106 104 106 102 95 85 90 99 95 90 92 100 103 108 109 111 115 116 123 125 128 129 119 103 83 65 54 50 59 62 68 79 87 98 114 128 131 138 145 168 178 190 190 184 180 156 115 103 100 89 69 60 49 55 35 60 148 84 112 149 111 74 87 96 84 144 195 201 202 209 217 219 215 218 220 216 200 152 177 195 231 247 244 243 243 243 243 243 121 134 147 142 141 137 133 123 116 107 93 92 94 91 91 96 97 94 95 99 94 96 98 98 104 102 100 96 79 65 60 52 57 60 59 54 47 45 54 69 76 88 102 123 131 141 156 174 194 212 221 211 184 134 97 93 87 66 81 112 97 115 78 31 97 104 106 195 193 130 107 108 111 128 166 197 205 214 223 224 219 220 221 215 195 163 184 199 234 247 243 243 243 243 243 243 108 129 142 143 140 137 130 126 120 113 100 98 99 94 89 102 104 99 95 93 88 80 68 51 40 29 22 11 57 74 9 43 33 27 41 58 56 39 41 51 63 77 92 115 128 140 160 183 208 223 233 224 181 129 105 111 87 77 107 127 108 115 128 109 119 102 149 215 183 164 183 184 176 178 180 194 212 223 224 227 226 225 219 215 199 173 185 210 238 246 244 244 243 243 243 243 108 131 142 143 141 136 130 125 107 100 104 102 94 85 93 99 104 99 84 69 58 36 25 35 30 44 50 10 43 93 63 113 126 69 33 42 63 67 40 48 61 74 89 112 127 140 156 182 213 224 233 226 205 184 176 165 96 91 102 117 130 130 145 154 160 153 184 191 189 219 217 205 215 212 203 207 214 221 227 228 225 223 221 219 202 176 189 218 240 244 244 244 243 243 243 243 110 131 145 144 138 133 128 126 111 98 102 90 85 94 95 101 94 70 55 46 32 21 46 86 75 73 89 64 58 82 75 130 129 122 89 32 55 106 91 57 57 75 90 113 126 138 155 181 201 220 230 225 216 214 203 189 152 116 100 113 127 137 151 159 170 184 188 200 206 206 204 214 221 221 219 215 219 222 227 228 225 222 221 218 201 177 195 225 243 244 243 243 243 243 243 243 121 129 142 142 139 133 129 128 124 120 105 86 94 102 89 86 77 63 52 51 57 76 91 98 101 80 83 85 87 117 107 119 115 111 106 95 86 118 128 94 64 73 89 110 122 136 154 178 192 212 226 228 218 218 181 162 151 115 96 102 112 129 145 151 166 185 190 192 195 204 215 223 224 222 224 223 225 225 227 229 227 224 219 215 202 180 197 227 244 245 245 242 243 243 243 243 124 133 143 144 141 136 133 134 135 128 124 123 119 110 102 89 78 76 87 106 114 111 106 103 110 108 105 110 115 111 121 125 124 113 114 118 122 133 124 104 72 74 92 111 124 135 150 171 189 208 226 226 220 214 201 160 144 120 100 98 111 115 111 126 142 154 169 183 197 203 212 213 219 224 228 229 225 223 225 226 226 223 217 215 201 182 203 229 243 245 245 243 242 243 243 242 133 143 147 146 144 139 136 138 137 132 134 133 125 115 107 90 90 101 114 112 114 116 114 115 114 113 111 111 113 119 124 126 115 101 101 109 121 126 111 100 89 88 102 115 123 130 148 168 186 206 226 224 225 216 212 189 159 151 134 117 123 131 128 131 135 148 167 182 197 206 214 218 223 225 227 229 228 229 228 224 225 222 217 215 198 183 207 231 243 245 244 243 242 243 243 241 149 148 146 146 146 140 134 134 134 131 131 133 130 116 109 111 116 118 120 120 112 106 108 109 112 114 114 116 118 125 124 117 107 103 111 121 126 120 106 107 102 100 110 115 118 124 138 159 183 204 224 226 227 223 217 211 187 174 170 171 173 177 174 173 175 178 184 195 210 215 214 215 218 221 225 228 231 232 228 226 225 223 219 215 196 181 208 232 243 244 243 242 243 243 241 240 156 151 149 147 145 139 134 133 132 132 131 133 132 131 128 127 126 129 130 126 116 110 100 92 93 92 88 89 91 91 91 97 110 115 123 130 126 120 112 114 107 107 113 112 115 123 133 149 179 203 223 226 227 226 223 224 222 196 180 177 177 189 194 195 201 202 202 209 222 229 228 226 221 218 220 222 225 230 229 224 221 222 218 216 196 179 209 233 244 243 242 243 243 241 239 238 164 155 150 149 148 142 139 137 133 132 131 136 137 137 135 133 133 131 127 125 119 111 104 101 92 90 92 95 98 111 122 125 130 131 133 135 128 118 115 115 113 118 118 115 114 125 132 145 178 205 222 225 226 227 224 226 229 222 201 177 173 176 187 193 197 205 214 218 217 224 228 221 220 218 222 225 229 232 230 224 220 221 219 214 196 191 214 235 244 243 242 242 242 241 239 237 170 162 154 151 145 145 142 136 133 131 132 135 136 134 136 137 136 135 132 127 122 120 114 116 117 124 131 131 131 136 137 136 133 135 142 138 120 119 124 120 115 124 125 117 115 123 127 141 171 198 214 221 224 225 225 227 226 231 223 201 188 174 177 186 189 195 208 216 216 217 219 218 219 217 219 227 233 233 230 225 223 220 218 212 210 213 217 237 244 243 242 241 242 241 239 236 175 166 154 152 146 147 143 138 131 132 135 136 137 138 140 140 138 139 140 141 137 137 135 134 131 134 136 132 130 133 132 137 139 140 141 129 120 124 127 125 120 126 126 116 114 120 129 141 165 192 206 216 220 223 223 225 226 229 229 223 207 199 183 183 191 194 197 207 208 211 210 214 219 220 222 230 237 234 230 224 221 218 214 213 219 213 212 239 244 243 242 241 241 240 237 234 175 169 158 154 149 146 141 141 133 131 134 137 135 138 140 139 140 141 141 142 138 138 137 137 134 137 138 133 130 133 133 136 141 140 133 126 127 126 124 123 127 124 118 114 116 122 130 139 165 191 207 217 218 222 223 224 225 228 225 216 216 217 203 195 197 200 199 206 206 213 214 215 219 224 231 236 238 234 230 226 222 217 212 214 221 213 216 239 244 243 242 241 240 239 236 233 173 171 164 153 149 152 146 144 134 134 136 134 136 135 136 141 141 141 141 143 142 143 141 142 139 138 140 137 136 137 136 142 140 134 127 131 132 130 127 123 128 125 120 117 117 126 131 138 162 185 206 217 217 224 227 228 227 230 225 220 221 214 210 210 206 202 201 204 209 214 209 213 225 232 236 238 235 231 227 226 222 217 214 213 222 213 218 241 243 243 242 241 239 238 236 232 172 172 165 155 150 155 150 147 140 134 137 136 138 137 141 141 141 144 141 146 145 146 145 145 143 144 143 143 141 141 141 140 136 134 131 133 133 130 129 126 128 128 125 121 122 131 132 143 155 181 206 215 216 222 229 233 228 230 233 228 225 218 214 211 209 207 206 207 208 209 211 217 226 235 237 233 232 230 226 223 221 216 214 213 219 214 219 241 242 243 243 241 239 236 235 231 169 170 165 157 155 158 152 144 139 134 137 138 137 137 140 140 141 144 142 144 147 144 142 148 145 148 146 139 140 141 138 137 135 135 141 134 131 133 129 122 122 125 121 122 127 131 133 142 155 178 202 219 226 223 230 230 226 219 218 231 228 222 218 214 211 210 207 209 210 210 220 225 227 234 236 234 229 228 226 223 222 218 212 212 218 213 219 240 243 243 241 240 237 234 233 229 165 167 164 158 159 158 156 146 140 135 139 139 138 139 137 137 138 140 142 141 145 141 139 148 147 149 142 137 143 141 139 137 137 138 137 137 138 132 113 108 116 120 123 122 128 137 133 135 155 182 203 228 235 226 226 223 222 219 195 204 218 221 220 219 219 216 212 214 216 217 225 227 229 234 237 233 230 228 227 224 223 218 212 211 216 211 221 241 243 242 240 238 235 232 229 225 162 164 163 160 159 157 160 151 142 137 139 137 138 138 134 133 136 137 140 138 139 140 138 143 146 145 145 146 142 139 142 141 139 140 141 141 137 120 98 109 121 122 121 119 132 137 134 136 155 177 203 233 234 227 226 219 216 223 200 175 194 204 215 219 223 218 213 216 219 222 224 229 230 232 232 231 230 229 225 225 223 216 212 211 212 210 224 242 242 241 239 237 234 230 227 223 162 165 165 160 156 156 159 154 141 140 136 137 137 134 134 135 138 138 137 137 140 143 145 145 145 147 148 143 139 139 141 140 138 139 141 143 132 107 104 118 126 125 122 126 138 135 136 138 153 177 204 234 237 226 228 220 213 218 217 174 160 184 200 204 213 217 212 213 216 220 223 225 227 231 232 230 229 228 223 222 222 218 212 211 210 212 228 243 242 240 238 236 234 229 226 221 162 166 164 158 157 159 162 159 144 135 135 134 134 134 134 136 135 135 136 139 142 144 147 147 145 146 149 144 138 139 139 140 141 141 142 139 118 85 109 129 129 128 131 134 141 136 138 139 152 177 209 233 243 231 229 228 223 221 222 194 144 155 174 189 202 210 212 211 212 216 223 224 226 229 229 227 227 226 224 223 218 217 215 211 210 211 232 241 242 239 237 235 232 228 225 220 155 165 163 158 160 157 160 159 148 135 133 131 133 132 132 136 138 138 137 138 138 139 142 143 144 144 142 142 143 142 139 138 140 141 139 133 89 74 117 134 135 136 134 132 133 128 132 134 141 158 184 212 221 219 223 227 227 224 223 208 160 135 154 170 187 199 210 211 208 210 216 220 223 225 224 224 223 223 223 223 216 216 214 209 202 206 235 241 240 237 236 233 230 226 222 217 146 163 163 158 157 151 156 158 150 140 131 130 133 130 130 132 136 136 133 133 134 136 140 142 141 141 141 139 138 139 140 141 140 138 136 109 61 78 119 129 131 129 126 120 122 117 116 116 118 136 158 186 204 206 211 218 215 211 216 213 184 144 141 156 172 188 200 206 206 208 211 212 218 220 220 223 222 219 220 221 220 216 212 202 188 215 238 242 239 236 235 231 228 224 220 215 135 161 159 157 154 148 153 161 155 141 129 128 130 131 130 133 134 131 131 133 134 136 140 137 139 136 140 139 137 137 139 138 139 135 130 77 49 78 113 116 117 112 104 102 103 108 103 104 105 115 134 166 189 195 170 146 158 181 197 210 207 168 139 150 166 178 189 199 203 205 206 209 217 221 222 224 224 220 219 218 220 213 209 198 205 234 241 240 238 237 234 230 225 221 217 213 124 154 153 156 154 148 148 156 156 141 130 128 127 130 130 131 131 128 129 132 131 130 136 136 134 134 135 134 135 138 141 137 135 134 115 60 49 69 102 97 89 73 47 38 44 71 89 94 92 101 123 156 182 121 52 45 77 132 182 208 212 190 154 142 160 177 187 195 198 201 205 208 216 219 221 222 220 221 217 217 219 214 210 217 233 241 240 238 238 236 232 228 224 220 216 211 112 143 155 156 153 147 144 151 155 143 130 121 124 129 128 126 129 127 127 130 129 129 133 137 133 135 133 130 131 133 135 135 130 133 109 61 61 63 79 84 67 33 17 20 31 47 68 87 84 80 121 181 201 187 143 98 91 151 203 219 222 207 181 152 150 166 184 197 199 202 206 208 215 218 220 216 216 221 218 217 217 214 211 228 240 242 240 239 238 235 230 227 222 218 213 209 102 128 157 152 155 150 144 152 153 146 134 123 123 126 125 124 128 127 128 129 130 133 130 134 133 130 130 131 134 129 127 132 133 138 128 78 66 63 63 63 52 35 36 52 64 64 66 76 78 72 127 173 193 219 214 180 191 212 221 222 223 214 197 173 155 159 174 191 200 203 204 207 211 214 217 218 222 220 215 217 214 211 210 226 237 241 239 239 238 234 229 225 221 216 211 206 91 115 157 151 152 149 145 149 154 147 139 130 125 125 121 124 125 123 124 126 127 131 129 128 132 132 129 129 131 126 130 135 137 140 143 118 100 93 81 66 56 51 53 54 61 60 56 62 72 84 122 167 186 191 199 209 218 220 225 224 221 216 206 191 172 164 175 184 193 204 209 211 213 216 216 213 218 217 216 217 216 210 211 231 239 240 238 237 235 233 229 224 220 215 210 205 68 83 147 155 151 145 144 147 153 147 142 135 128 124 124 123 124 125 123 124 124 128 130 128 130 130 130 130 128 124 131 131 131 142 140 137 134 127 116 105 89 78 71 75 71 76 86 95 90 95 130 162 174 182 192 191 196 218 222 221 221 219 211 195 189 177 176 186 189 200 206 208 211 215 215 212 211 214 216 216 216 207 209 232 240 240 239 236 234 231 229 223 220 214 209 205 108 137 161 147 149 144 143 143 151 147 141 131 129 125 124 120 122 124 123 125 125 127 129 131 131 130 131 130 133 132 133 137 142 145 139 137 136 135 127 119 116 113 97 95 100 102 112 118 125 108 122 143 168 182 172 178 193 211 222 221 222 222 219 197 182 182 187 190 195 200 200 200 204 211 213 211 207 213 217 215 214 208 214 234 239 240 238 236 234 230 227 222 218 213 208 204 245 255 214 147 145 146 145 144 149 146 143 138 131 125 121 123 121 121 123 126 125 126 128 129 130 132 128 128 133 130 131 139 141 140 136 140 136 130 128 115 104 98 91 93 103 114 131 143 153 132 116 139 165 160 152 161 175 181 197 208 219 217 217 214 197 181 185 188 197 200 197 197 203 209 210 212 214 212 216 217 215 207 213 234 239 239 237 235 233 229 225 220 217 211 207 203 255 254 239 163 142 145 146 148 151 144 141 137 132 120 123 122 120 120 122 124 126 125 127 129 126 127 127 127 125 130 133 137 139 136 133 134 129 130 118 101 91 89 86 100 105 128 148 171 159 148 135 132 149 150 150 149 151 151 179 187 191 194 188 194 200 189 185 190 199 205 199 196 201 207 211 211 216 215 217 217 214 209 220 236 238 238 237 235 232 228 224 220 217 211 206 201 253 253 254 184 138 147 145 147 150 143 141 138 133 122 121 122 119 120 121 119 119 121 122 124 126 126 125 128 126 128 134 136 134 135 129 122 117 115 99 89 94 96 86 94 113 129 163 185 182 160 144 140 151 153 150 137 144 152 173 176 172 178 174 176 185 189 193 197 204 213 202 193 198 207 210 208 212 217 217 215 212 207 222 237 238 237 236 234 231 227 224 220 216 211 205 200 254 253 255 211 140 145 145 147 147 142 141 138 130 121 120 123 121 120 118 114 117 121 120 121 124 126 123 120 123 124 134 127 127 120 108 101 93 87 83 80 95 98 91 100 115 140 185 196 193 173 157 159 173 177 165 153 164 165 169 176 163 164 169 185 169 154 180 189 200 213 208 196 197 208 206 206 212 217 213 213 212 209 221 236 238 236 235 234 231 227 224 219 215 210 204 199 253 252 255 236 152 140 146 147 146 142 141 139 132 128 124 124 122 116 112 114 122 119 117 121 121 125 120 120 123 126 126 132 116 103 100 83 73 72 72 89 106 108 112 122 133 153 177 190 195 182 173 173 182 187 190 187 188 187 184 179 156 154 163 164 162 158 164 173 187 212 215 203 193 200 208 208 213 215 212 210 210 207 222 236 237 236 235 233 230 226 223 218 213 208 203 197 252 253 255 251 166 134 146 148 145 143 143 142 135 128 122 127 127 123 119 117 117 117 115 116 121 119 115 122 124 124 124 110 92 80 70 71 74 77 84 101 123 135 131 133 126 118 125 127 147 162 161 153 148 149 161 162 177 197 203 196 176 171 170 156 150 144 161 169 179 207 217 207 197 199 209 211 213 214 213 210 207 206 225 238 237 236 234 233 230 226 223 217 213 208 202 197 255 255 254 255 194 132 138 144 143 142 144 142 139 129 126 129 125 120 117 112 110 116 117 116 119 114 114 120 122 116 101 91 80 75 78 74 89 92 100 121 127 117 99 90 86 83 94 94 97 114 126 127 125 140 145 151 150 146 152 164 171 179 178 158 147 135 154 166 182 199 215 212 200 200 204 209 212 212 211 209 205 207 229 237 236 236 234 231 228 224 220 215 210 207 201 196 255 255 254 255 204 122 129 141 142 142 144 145 141 133 129 127 124 121 119 118 117 113 114 117 117 116 119 122 118 99 84 85 90 89 76 83 100 108 98 92 82 75 78 85 86 84 77 68 72 90 96 96 84 102 106 121 127 119 123 132 127 123 127 107 117 139 143 167 191 203 214 212 202 200 201 207 210 211 210 209 203 210 233 237 236 236 233 231 228 224 220 214 210 206 198 193 255 255 254 255 221 128 127 136 141 142 142 145 144 133 126 123 123 120 119 121 120 114 115 117 115 117 120 122 112 97 87 93 99 87 78 92 93 80 65 56 61 62 59 55 58 60 57 55 60 68 64 78 71 73 91 111 111 118 119 124 95 90 104 77 77 112 124 157 176 201 215 214 205 203 203 204 208 209 209 206 204 220 238 235 235 235 233 230 227 223 219 214 210 206 199 194 255 255 254 255 235 147 119 134 143 141 139 142 146 135 126 124 123 118 117 121 118 118 120 116 116 116 119 121 110 106 91 97 102 97 82 50 37 33 35 38 40 47 54 55 70 80 103 120 132 139 124 136 153 156 164 193 215 225 197 180 177 194 187 153 111 98 115 135 171 197 214 216 210 207 204 202 207 209 209 205 198 218 238 235 235 235 232 230 227 223 219 212 206 203 198 193 255 255 254 255 242 166 130 135 143 142 140 139 145 137 128 123 119 122 120 119 119 115 113 116 118 117 119 119 112 104 102 107 101 95 82 74 69 60 55 61 68 80 93 103 106 112 125 140 146 164 164 158 162 172 199 228 233 217 210 215 220 227 214 193 160 142 136 140 171 198 212 217 212 208 204 204 207 209 204 200 198 216 236 237 236 235 234 231 226 222 219 212 208 202 197 193 255 255 255 255 244 177 140 125 136 143 136 143 146 137 126 120 118 119 115 114 117 113 112 113 116 116 119 118 107 103 119 113 94 98 111 124 130 124 113 109 105 99 99 102 101 106 116 137 151 173 177 157 171 191 203 209 199 204 217 221 217 226 227 210 189 173 162 164 176 199 212 216 212 208 205 203 202 206 200 194 205 225 236 236 236 235 233 230 226 222 218 213 209 202 196 191 255 255 255 255 249 182 143 127 123 128 137 139 141 139 126 117 117 116 113 113 114 111 111 115 116 113 120 123 120 116 111 113 108 108 131 123 126 129 127 131 122 115 112 108 105 106 109 127 140 159 171 164 176 180 183 192 205 215 216 222 222 224 222 219 200 181 178 173 179 200 208 212 211 208 202 200 202 198 195 194 214 234 236 235 234 234 232 229 226 221 217 212 207 200 194 189 255 255 255 255 253 190 151 139 123 124 134 130 134 142 129 115 114 114 115 114 116 115 110 112 115 113 118 126 125 118 110 109 119 125 126 131 131 125 124 128 127 127 131 131 124 120 117 123 127 136 149 165 159 168 180 197 211 210 215 218 223 221 218 214 194 179 182 174 176 203 206 209 209 205 200 202 199 196 192 196 221 235 236 234 233 233 231 228 225 221 217 211 205 199 194 190 254 255 254 252 254 202 148 138 120 120 130 123 130 138 128 116 113 109 110 111 114 116 111 110 114 114 116 120 121 116 122 120 116 127 130 129 128 124 120 121 122 125 127 125 128 123 121 123 122 118 122 148 147 158 178 193 203 208 211 211 215 212 216 211 202 187 171 180 185 198 207 206 209 202 199 201 197 196 189 199 225 234 234 234 233 232 230 227 224 220 215 209 204 198 194 188 255 254 252 253 255 212 145 144 126 113 119 121 122 127 129 122 111 107 106 108 111 113 110 106 111 116 114 115 119 119 118 126 122 123 123 120 126 121 120 121 117 123 120 117 119 122 129 139 133 122 117 135 140 147 167 188 195 205 213 215 209 207 214 216 215 194 172 187 193 195 201 195 198 200 201 199 197 198 193 215 235 235 234 234 232 232 230 226 223 219 214 208 203 198 192 187 252 251 255 252 215 185 155 140 128 115 112 116 118 119 126 119 111 102 105 108 107 114 111 107 110 110 110 114 117 116 111 111 118 116 116 126 126 117 118 116 115 121 114 113 106 110 121 139 132 139 139 129 124 143 150 171 191 194 208 215 204 204 210 206 197 188 182 193 197 196 199 190 187 198 196 196 198 189 197 226 234 232 234 233 231 230 228 225 221 217 213 207 203 197 192 186 252 255 240 157 125 195 155 140 134 119 105 106 114 114 119 118 114 107 105 105 107 111 107 105 110 112 114 115 112 113 112 108 116 120 120 122 123 124 126 125 113 112 115 106 103 108 111 132 118 93 101 110 110 135 138 157 182 190 203 213 208 203 199 203 202 195 185 181 188 194 203 196 194 201 199 199 195 188 215 232 232 230 233 231 230 229 227 224 220 216 212 206 202 196 191 185 255 235 134 40 149 227 160 142 136 122 106 94 103 109 112 119 116 113 102 97 103 105 107 108 108 110 114 109 103 95 99 111 114 114 126 123 116 119 127 127 116 115 115 106 108 111 114 125 104 85 100 100 113 139 147 167 185 200 209 210 210 212 208 208 198 189 192 196 202 203 200 196 196 202 203 197 185 200 229 230 229 229 230 229 229 228 226 223 219 215 210 206 201 196 190 185 239 141 26 59 216 229 171 139 136 122 110 99 95 100 104 109 113 113 104 100 101 100 103 104 103 106 106 104 104 99 107 111 114 119 124 125 122 127 131 123 123 128 123 121 120 123 120 118 116 112 119 121 141 160 169 187 199 204 208 212 218 214 209 207 200 187 188 180 186 203 199 197 195 195 197 192 190 219 232 227 228 229 230 229 229 227 224 221 217 214 210 204 200 194 189 183 159 49 7 132 232 226 180 140 132 124 116 104 95 96 100 103 109 111 107 102 101 99 98 100 102 101 101 103 106 102 103 111 116 120 121 120 124 129 126 129 133 131 133 134 134 136 138 136 135 147 146 151 173 186 199 201 202 208 212 215 223 214 201 191 194 176 175 182 192 201 198 196 196 196 186 189 212 226 230 227 228 228 228 228 228 226 223 220 217 213 209 203 198 193 187 182 67 12 32 183 235 227 188 137 128 121 114 104 95 89 90 95 97 101 103 98 97 96 95 98 98 97 100 104 102 104 101 111 113 113 120 120 126 122 124 135 138 136 138 139 139 148 153 146 152 169 167 167 174 178 192 203 214 211 218 222 222 215 208 193 197 201 189 185 197 197 200 200 193 187 182 204 227 228 229 229 228 228 227 227 227 225 222 219 216 213 209 203 199 194 188 183 26 2 75 215 230 229 203 145 124 118 113 105 95 84 84 85 91 93 102 100 96 93 90 93 96 95 93 100 100 100 101 105 108 115 121 117 118 123 130 128 137 141 141 137 142 143 150 153 161 164 159 157 155 166 179 189 201 209 216 216 212 206 208 198 196 195 187 187 192 196 191 198 186 176 202 224 228 230 230 228 228 227 226 225 225 224 221 219 216 212 208 203 198 193 188 182 17 3 115 223 225 225 212 155 121 114 112 104 95 87 86 79 86 89 99 95 95 94 89 88 87 91 94 96 97 96 90 99 115 118 118 110 109 124 125 122 127 132 134 131 126 115 141 151 160 163 156 151 150 167 184 187 185 195 204 207 217 214 197 178 176 184 193 192 197 194 192 188 177 199 221 227 226 227 227 227 226 226 225 224 224 223 221 219 215 211 207 202 196 191 186 180 11 15 144 222 225 219 217 169 123 112 109 104 97 88 86 83 75 76 86 88 93 92 89 83 82 91 87 84 92 90 88 105 109 113 109 102 106 107 115 117 104 115 115 108 110 115 134 151 168 159 164 166 147 153 165 176 182 184 188 192 196 197 187 168 150 180 200 192 196 194 183 172 186 219 224 226 226 227 227 226 226 225 225 224 223 223 221 219 215 210 207 202 197 191 186 180 3 29 166 220 224 218 217 187 131 108 101 99 96 87 84 82 78 71 75 84 89 88 86 86 84 85 87 81 80 77 82 93 87 96 93 97 102 107 116 98 92 97 95 99 101 114 131 143 157 162 156 144 152 154 152 171 182 172 169 173 162 168 183 178 171 178 183 184 194 184 165 184 210 225 222 224 226 226 226 225 225 225 224 223 222 222 219 218 215 211 207 202 196 190 185 178 2 43 174 218 219 219 216 203 149 110 96 92 94 86 83 80 79 78 74 71 82 86 87 87 81 76 77 79 76 75 74 64 78 105 107 98 82 91 103 97 95 88 85 86 100 124 132 137 142 159 165 151 146 148 157 168 174 164 159 165 165 164 181 186 201 187 169 166 181 163 180 204 180 208 225 223 224 226 225 225 225 224 224 223 222 222 220 218 215 210 206 201 195 190 184 178 1 47 172 213 218 218 215 208 173 114 98 91 92 89 86 79 75 76 78 67 69 74 86 86 75 75 74 75 70 70 67 65 84 101 100 81 75 82 88 95 79 81 88 94 102 104 119 134 149 157 162 149 133 134 148 167 165 173 169 157 156 165 163 166 187 175 154 155 160 183 207 198 125 159 224 226 222 227 224 226 225 224 223 222 222 221 218 217 214 209 204 199 195 189 184 178 1 44 164 206 214 215 214 206 197 143 95 92 89 88 87 82 77 74 77 74 71 70 66 67 73 78 80 73 70 73 68 67 68 69 79 90 86 75 65 73 82 85 88 84 88 104 116 126 140 142 136 131 128 128 134 144 152 164 161 159 152 166 172 167 162 148 142 156 181 201 208 221 129 96 196 225 221 226 225 225 224 223 222 221 221 220 217 215 213 208 204 200 195 189 183 178 0 31 138 195 208 212 212 207 202 183 119 88 85 81 83 83 79 73 74 75 73 71 62 53 55 63 67 66 68 74 69 66 60 56 69 75 76 71 62 79 71 66 77 80 86 104 107 110 117 124 132 125 113 122 134 148 154 151 144 152 159 150 172 178 153 132 141 177 204 205 209 251 173 58 138 215 227 224 224 224 224 222 222 222 222 221 218 215 211 207 205 199 193 188 183 177 1 18 112 183 201 205 206 206 199 195 162 102 88 82 79 82 82 74 71 72 69 71 72 65 55 46 50 55 59 65 66 63 64 68 59 62 76 73 65 69 55 60 72 78 73 92 93 107 110 117 118 108 104 118 142 143 143 145 140 130 142 146 163 167 136 142 175 202 205 205 219 255 196 53 83 178 224 226 223 224 223 222 221 221 220 219 217 214 211 207 205 199 193 188 183 177
## 2788                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         156 157 159 163 171 174 165 156 142 134 173 209 209 210 195 178 180 185 181 174 164 152 153 163 170 175 177 174 172 174 176 179 179 165 152 131 110 116 122 147 175 186 193 179 156 149 150 155 165 166 156 149 145 143 149 153 149 145 140 136 137 138 134 129 125 121 130 143 158 176 180 175 165 148 135 125 115 100 86 67 51 40 30 29 29 32 36 36 35 33 31 32 33 33 33 33 158 157 155 157 166 171 164 156 142 133 170 204 205 207 193 178 180 184 180 173 163 151 152 161 167 172 174 173 172 173 175 178 177 165 153 133 112 115 118 141 166 179 190 177 155 148 148 152 161 162 153 146 142 139 145 149 146 143 139 136 137 138 134 129 124 119 126 137 150 167 172 169 161 147 136 128 120 105 91 72 55 44 34 32 31 33 36 36 35 33 31 32 33 33 33 33 162 157 145 142 155 162 161 157 141 130 161 190 194 199 189 179 180 183 177 170 161 150 149 156 160 164 167 170 172 172 172 174 173 165 157 137 116 112 109 125 144 162 182 173 153 145 141 144 151 152 145 140 133 129 133 137 138 139 137 135 136 137 133 128 121 113 116 122 131 144 150 153 151 144 140 137 131 116 102 83 67 54 43 39 36 37 37 37 35 33 31 31 33 33 33 33 164 157 138 132 147 156 158 157 141 129 154 178 184 190 185 180 179 180 174 167 159 149 148 152 154 157 161 168 172 170 170 169 168 164 159 141 121 113 104 114 126 146 169 166 152 144 139 139 143 143 137 132 126 122 125 128 132 135 134 133 134 135 131 127 119 111 110 112 117 127 134 140 143 142 142 141 137 123 110 93 77 64 52 46 41 40 38 37 35 33 32 32 33 33 33 33 163 155 133 126 142 153 154 152 140 131 147 162 170 178 178 178 177 175 169 162 156 148 146 147 149 150 156 166 172 168 165 161 159 160 160 146 131 118 104 107 112 127 145 150 150 148 143 138 133 129 124 121 119 117 118 120 124 128 128 128 130 131 129 126 120 113 110 108 109 114 119 127 133 139 141 137 132 124 115 101 89 77 65 56 48 44 39 37 35 34 34 34 33 33 34 34 162 153 129 123 140 152 150 148 139 132 141 149 159 168 173 176 174 171 165 158 153 148 145 144 144 145 152 165 172 166 161 154 151 157 161 150 140 122 105 102 101 111 124 136 149 151 148 139 126 118 114 112 113 114 114 114 118 122 122 122 124 127 128 128 124 120 115 109 107 107 110 118 125 135 138 132 127 123 118 108 99 87 75 65 53 46 40 37 35 36 37 36 33 33 34 34 157 150 131 126 143 154 148 142 137 134 136 139 149 159 167 174 170 165 159 153 150 147 144 142 142 143 150 164 171 164 156 147 141 151 159 152 147 127 107 100 95 98 104 123 150 157 155 142 120 108 106 106 110 113 112 112 115 117 112 107 114 121 129 137 139 141 136 128 121 114 112 115 119 127 129 122 117 118 117 113 108 97 85 72 57 48 39 38 36 38 40 39 35 34 34 34 152 147 132 130 145 154 144 137 134 133 131 131 141 151 161 171 167 161 156 150 147 146 144 141 139 140 146 163 171 163 154 141 133 144 154 152 152 132 112 101 91 89 88 110 146 157 157 144 118 104 102 102 107 110 108 107 109 111 103 96 105 114 126 139 146 153 149 141 132 121 115 113 115 121 122 115 112 115 117 116 115 105 94 79 62 52 41 39 37 40 42 40 35 34 34 34 140 138 134 133 139 142 135 129 124 121 122 123 134 144 156 167 167 165 160 153 148 143 141 139 135 131 137 159 170 168 162 143 129 132 134 142 150 140 128 113 96 89 83 95 117 130 139 138 129 121 114 107 100 95 91 90 93 96 95 94 97 100 108 116 124 132 134 133 130 124 121 118 119 122 123 123 122 122 120 118 115 109 101 89 76 64 51 45 39 37 37 35 32 31 31 31 128 130 134 135 134 132 127 122 115 110 114 117 128 137 150 162 166 169 164 155 148 141 138 137 132 124 129 154 169 171 167 145 127 121 118 133 148 145 141 123 101 89 78 82 92 105 122 132 137 135 125 112 95 83 77 75 79 83 88 92 90 88 92 95 104 113 120 125 127 126 125 122 121 121 123 128 130 126 122 118 115 112 107 99 89 75 60 51 41 36 32 30 29 29 29 29 123 124 128 130 130 130 128 124 116 110 112 114 123 131 143 154 160 164 160 153 146 138 134 132 127 120 124 144 157 160 159 145 132 127 123 131 139 136 133 120 106 92 77 78 86 95 108 118 125 126 121 113 100 89 83 79 81 82 84 87 84 80 82 83 90 97 103 109 110 110 111 110 110 112 113 115 114 113 112 113 112 113 111 104 97 83 69 58 47 41 37 35 34 33 32 32 117 118 121 124 127 129 128 126 117 110 111 111 118 125 136 147 153 159 156 151 144 136 131 128 123 117 119 134 145 150 152 145 138 132 127 129 130 128 125 118 110 95 76 75 80 86 95 104 113 118 117 114 104 95 88 83 82 82 81 81 77 72 72 72 77 83 88 93 95 96 97 98 100 103 104 102 100 101 103 107 110 113 115 110 105 91 77 65 53 47 42 40 38 37 35 34 112 113 116 118 122 125 126 125 117 111 110 107 111 115 125 136 143 151 150 147 141 133 127 122 117 113 114 122 129 136 140 141 140 135 130 126 121 120 117 116 115 100 80 76 76 78 83 90 99 106 111 112 107 101 95 89 86 84 82 81 75 69 66 64 66 69 72 75 77 78 80 82 85 89 90 86 84 85 87 92 97 102 106 103 101 90 79 70 60 56 53 49 42 38 35 33 106 107 110 112 117 121 123 124 118 113 108 103 104 106 115 125 134 142 144 143 138 129 122 115 111 109 109 111 114 122 129 136 142 138 134 123 112 111 109 114 120 106 85 77 72 70 72 77 86 95 104 110 110 108 102 96 91 87 84 81 73 65 60 55 55 55 56 58 60 61 63 66 70 75 76 71 68 70 72 78 84 91 97 96 96 89 81 75 68 66 64 58 47 40 34 32 106 107 109 111 114 116 117 118 116 114 108 101 100 98 106 114 124 134 137 139 134 126 119 111 107 107 106 104 105 111 117 127 134 132 131 121 112 111 109 114 120 109 92 84 76 72 69 72 78 86 96 103 106 107 103 99 95 91 89 86 78 69 64 57 54 53 52 51 51 51 52 55 58 63 65 62 60 60 59 63 67 72 78 79 80 77 74 73 71 73 75 67 55 47 39 36 106 107 108 109 110 111 110 111 113 115 107 99 96 91 97 103 113 125 130 134 131 123 115 108 104 104 102 98 96 100 106 116 124 126 127 119 112 110 108 113 119 112 100 91 81 75 67 67 71 77 88 96 103 106 104 103 99 96 94 92 83 74 67 60 55 51 48 45 43 41 41 43 47 52 53 54 53 49 47 47 48 53 57 59 62 64 66 70 75 80 86 78 65 55 45 41 108 109 110 110 109 107 103 102 109 114 106 99 93 87 92 97 106 117 123 127 126 120 113 105 102 104 102 97 95 95 98 105 111 115 119 117 115 113 112 113 115 111 105 97 88 79 70 66 67 70 77 84 92 96 97 98 97 95 93 92 84 76 70 62 58 54 50 46 42 39 38 40 42 46 48 49 48 44 40 38 38 41 45 49 52 55 59 66 73 80 88 83 75 66 54 50 110 111 112 112 107 103 96 93 104 112 105 98 91 84 87 90 99 108 115 120 120 116 111 103 100 103 102 98 94 91 90 93 96 103 110 115 118 117 116 113 110 109 109 103 95 85 74 67 63 62 65 71 79 84 89 91 93 93 91 90 84 77 72 65 61 58 53 48 43 37 36 37 38 41 44 45 44 38 33 30 27 30 33 38 42 47 52 61 71 79 89 89 86 78 64 59 109 110 112 112 107 101 93 89 100 109 103 98 91 84 85 86 92 100 107 113 114 112 108 102 100 102 101 98 94 89 85 85 86 94 101 109 117 117 116 110 104 105 108 104 97 88 78 69 62 58 58 61 68 74 79 83 85 87 84 81 77 71 67 63 60 58 54 49 44 38 36 36 38 41 43 42 39 35 31 29 27 31 34 39 42 46 51 58 66 74 83 85 86 78 65 60 108 109 113 113 106 100 91 85 95 104 101 97 91 85 83 81 86 91 98 104 107 107 105 101 99 100 99 97 94 87 81 77 75 83 92 103 115 115 114 106 97 100 105 103 99 92 83 73 61 54 50 50 55 61 67 73 76 78 73 68 66 62 61 59 58 56 54 50 45 39 37 37 39 42 42 38 34 32 29 30 30 35 39 43 46 48 51 56 61 68 74 78 82 76 63 58 104 106 110 111 105 100 92 86 93 99 98 96 92 87 83 79 82 85 90 97 100 102 102 100 98 98 96 95 93 86 80 75 72 78 85 95 107 107 108 101 93 95 100 100 98 93 86 76 63 55 49 48 52 56 61 66 68 70 64 59 57 55 55 54 54 54 52 48 43 38 36 36 38 42 42 37 33 31 29 31 32 38 42 46 48 49 50 53 58 62 67 70 73 68 57 52 98 100 105 106 104 101 94 90 91 92 93 95 93 89 84 78 77 78 82 88 93 96 97 98 96 93 92 92 91 85 81 76 72 75 78 86 94 96 98 94 89 91 93 94 96 93 88 80 68 59 51 49 51 54 57 59 61 61 55 49 48 47 48 48 49 50 48 45 41 36 35 35 37 42 42 36 32 31 30 32 34 40 45 48 50 50 49 51 53 56 60 60 61 55 46 42 93 95 101 103 103 102 97 94 90 86 90 94 93 91 84 77 75 72 76 82 87 92 95 96 95 90 88 88 88 84 82 78 74 75 75 79 83 86 88 88 87 88 88 90 94 93 90 83 72 63 56 52 54 55 56 57 57 56 50 44 44 43 44 45 46 48 46 42 38 33 32 33 36 41 41 37 33 32 31 32 34 40 44 47 49 48 47 48 50 53 56 55 52 47 40 37 87 89 96 100 102 103 102 100 89 81 87 93 92 92 85 76 72 67 70 76 82 89 93 95 94 87 84 83 83 83 83 81 78 76 73 71 71 73 77 81 86 85 83 86 91 92 92 87 77 69 63 59 60 59 58 56 54 52 47 43 43 42 43 44 44 45 43 39 35 30 29 30 33 38 40 38 35 33 32 32 33 37 41 44 45 44 43 45 47 51 55 52 45 40 35 34 85 87 93 96 100 101 102 100 89 80 85 91 91 92 85 77 72 66 67 69 75 82 87 90 90 84 80 79 78 79 80 80 79 76 73 69 66 68 71 76 82 82 81 83 87 88 88 85 78 73 68 64 62 60 57 54 51 49 45 40 40 38 38 39 40 42 41 37 34 29 28 28 31 36 38 37 35 33 32 33 35 38 43 45 46 44 42 43 44 47 51 47 41 37 34 33 84 86 90 91 94 95 97 96 88 81 84 88 88 89 84 80 74 67 64 61 65 69 74 80 82 79 77 74 71 72 73 75 76 77 76 72 69 69 68 70 73 77 82 82 80 78 78 77 76 74 72 68 62 57 52 49 48 47 42 37 34 30 30 31 34 37 38 36 33 28 27 27 29 34 35 34 32 31 31 35 39 44 50 52 53 48 44 42 40 41 43 41 39 38 37 36 83 84 86 88 90 91 93 93 87 83 84 86 85 86 83 80 74 67 62 57 58 61 65 70 73 75 73 70 67 67 68 71 74 75 76 72 70 68 67 67 69 73 79 77 73 71 71 72 76 76 76 72 64 58 52 48 49 48 43 37 33 27 27 27 31 34 35 35 33 28 27 26 28 33 34 32 30 30 31 36 41 47 53 55 56 50 45 41 38 38 38 38 38 38 38 38 79 80 82 83 85 86 88 89 86 84 84 85 84 83 80 76 72 65 61 56 55 55 58 60 63 68 69 66 64 63 64 67 69 70 70 68 65 66 66 68 70 70 69 67 65 65 68 72 78 81 81 78 71 65 59 55 56 55 50 44 38 31 29 28 30 33 34 34 32 29 28 27 28 32 33 31 30 31 32 37 42 48 53 54 55 50 45 42 39 39 39 39 38 38 38 38 76 77 78 80 82 83 85 86 85 84 83 83 82 80 77 74 70 65 61 56 54 53 53 53 56 62 64 62 60 59 59 62 64 63 63 61 59 60 61 65 68 65 60 59 59 62 67 73 81 86 87 85 79 73 67 63 63 62 57 51 43 35 32 28 30 32 33 34 32 29 28 27 28 30 31 31 30 31 32 37 42 48 53 54 55 50 45 42 39 39 40 39 38 38 38 38 74 75 76 77 78 79 80 81 81 82 81 81 79 76 75 72 70 67 65 61 58 55 53 51 51 54 55 53 51 50 50 52 53 50 48 46 43 45 48 53 59 56 51 51 54 61 69 78 89 95 98 98 92 87 82 78 76 74 67 60 51 42 36 29 29 31 32 33 33 31 30 28 28 28 29 30 31 31 32 38 43 49 54 55 56 51 45 42 39 38 39 38 37 37 37 37 72 73 74 75 76 76 77 77 78 79 78 78 77 74 73 71 70 69 68 66 62 58 54 51 48 48 48 46 45 44 44 44 44 40 36 33 30 33 37 43 50 48 44 47 52 62 73 85 97 105 109 109 104 100 95 91 88 85 76 68 58 48 39 31 29 30 31 33 33 32 31 29 28 27 28 30 32 31 32 38 43 49 55 56 56 51 45 42 39 38 39 38 37 37 37 37 72 72 73 74 74 74 75 75 74 72 74 75 75 74 73 72 71 71 71 69 66 62 58 54 50 46 44 43 43 43 42 41 40 34 29 25 21 22 25 31 38 41 43 50 58 70 84 97 110 119 123 125 120 116 110 105 101 98 87 76 66 55 45 34 31 29 30 32 34 34 32 30 28 26 27 31 33 32 33 38 43 50 56 57 57 52 46 43 40 39 40 39 37 37 37 37 72 72 73 73 72 71 72 72 69 66 70 73 73 74 73 72 72 72 72 71 69 64 60 55 50 44 41 41 41 42 41 40 37 30 24 19 14 15 16 22 29 37 44 53 64 77 95 108 122 131 136 138 134 129 123 117 113 109 96 84 73 62 50 38 32 29 30 32 34 35 33 30 28 26 26 31 33 33 34 38 43 50 56 57 57 52 46 43 40 40 40 39 37 37 37 37 70 70 70 70 68 68 67 67 64 63 67 71 72 74 72 69 68 67 67 66 64 59 55 49 45 43 42 43 43 45 45 43 41 35 29 23 17 18 20 25 33 42 52 63 75 89 106 119 132 139 143 145 141 136 130 124 119 114 102 91 80 69 56 43 36 30 30 32 33 35 34 31 29 25 25 30 33 33 34 38 43 49 55 56 57 53 47 44 41 41 41 40 38 38 37 37 69 69 68 67 65 64 63 63 60 59 64 70 72 74 71 67 65 63 62 62 59 55 50 43 41 42 43 44 45 47 48 47 45 39 33 27 21 22 24 29 36 47 61 73 86 100 117 129 141 147 150 151 147 143 136 130 125 119 108 97 86 75 62 48 39 31 30 32 33 35 35 32 29 25 25 30 33 33 34 38 43 49 55 56 57 53 48 45 42 41 41 40 38 38 37 37 66 66 64 62 60 59 58 57 57 58 63 68 69 69 66 61 60 60 60 61 58 53 46 38 36 39 41 41 41 43 44 45 45 41 38 33 29 31 33 38 45 57 70 82 96 109 125 135 144 148 149 148 144 141 134 128 123 117 108 99 90 80 68 54 44 35 33 33 33 35 35 33 30 26 26 31 34 34 35 39 44 50 56 57 58 53 48 45 42 42 42 41 39 38 38 38 64 63 60 58 56 55 53 52 55 57 62 66 66 65 61 56 56 57 59 61 58 51 43 33 31 37 40 39 38 40 41 43 45 43 42 39 36 39 42 47 54 66 80 92 107 119 133 141 147 149 148 146 142 139 133 127 122 116 108 101 93 85 73 60 49 38 35 33 33 35 35 34 31 26 26 31 34 35 36 40 44 50 56 57 58 53 48 45 42 42 43 41 39 38 38 38 64 63 60 57 55 53 51 50 53 55 58 60 59 57 55 54 60 67 74 81 80 73 63 50 45 44 43 40 38 38 38 40 42 42 42 41 41 45 49 55 62 75 89 102 116 126 138 144 147 147 144 142 138 135 129 124 120 114 108 101 94 88 77 65 53 41 36 33 33 35 35 34 32 28 27 31 34 35 36 40 44 50 56 57 58 53 48 45 42 42 43 41 39 38 38 38 64 63 59 56 53 51 49 48 50 52 53 54 51 48 50 52 64 78 91 103 103 96 85 68 59 52 47 41 37 35 34 36 38 40 41 44 46 51 56 63 71 84 99 112 125 134 144 147 146 144 140 137 133 130 125 121 117 112 107 101 96 91 81 70 58 44 38 33 34 36 36 35 33 30 29 32 34 34 35 40 45 51 57 58 58 54 49 46 43 42 43 41 39 38 38 38 66 65 61 58 54 52 50 48 48 48 48 46 44 41 46 53 71 92 110 129 131 124 112 94 80 67 57 50 44 40 37 38 39 42 44 47 50 57 63 71 79 92 107 119 131 139 147 149 146 143 138 135 132 129 125 121 117 112 107 100 96 93 83 73 60 46 39 33 34 36 36 36 34 31 30 32 34 34 35 40 45 51 57 58 59 54 49 46 43 42 43 41 39 38 38 38 69 67 63 60 56 54 52 49 46 44 41 38 36 34 43 55 79 108 131 157 161 155 142 122 104 83 69 60 53 45 40 40 41 44 47 51 55 63 71 79 88 100 115 126 137 144 151 152 147 142 136 132 130 128 125 122 117 112 106 99 97 95 86 76 63 49 41 34 33 35 36 37 35 32 31 33 35 35 36 41 46 52 58 59 60 55 49 46 43 43 43 41 39 38 38 38 72 70 65 60 53 48 43 38 37 37 35 33 30 28 37 49 75 107 133 164 171 169 158 139 123 108 96 87 78 69 62 60 59 62 65 70 75 82 89 95 101 112 124 132 138 141 145 144 140 136 131 127 125 123 120 117 114 110 108 104 97 91 83 74 66 57 50 42 37 32 31 32 33 35 37 37 37 36 36 41 46 54 61 62 62 56 49 44 40 40 41 40 38 37 36 36 76 73 67 60 50 40 31 23 26 28 28 27 24 22 30 40 68 101 131 166 177 180 172 154 144 137 130 120 111 100 91 86 82 85 88 94 100 105 112 114 118 126 135 138 137 136 135 134 130 128 125 121 119 116 114 111 110 108 110 110 98 86 79 72 69 67 61 52 42 29 25 25 30 40 45 43 40 37 36 42 47 57 66 65 65 57 48 42 36 37 39 38 36 35 34 34 76 73 66 59 48 37 28 21 24 26 27 28 26 24 31 40 66 97 127 160 172 178 172 156 148 145 140 133 127 118 112 109 107 110 112 117 122 125 128 129 131 134 139 139 137 136 133 131 128 126 122 119 117 114 111 109 108 108 108 107 96 85 80 75 74 74 68 57 45 30 26 27 35 48 56 56 54 50 48 51 53 60 66 64 63 55 47 41 35 35 37 37 36 36 35 35 74 71 64 57 46 36 28 22 25 27 29 32 31 30 36 42 65 92 120 151 164 170 167 153 147 146 143 141 138 135 133 134 135 138 140 144 146 146 146 145 144 142 140 139 138 137 134 131 128 125 121 118 116 113 110 108 107 107 104 101 94 86 83 80 80 81 73 61 48 33 30 34 43 59 70 73 75 70 66 65 63 64 65 62 58 51 45 39 35 34 35 35 37 38 37 37 71 68 61 55 44 36 29 24 26 28 31 35 36 36 41 46 66 90 116 145 158 167 165 154 148 144 141 142 142 142 143 145 147 150 152 153 155 153 151 149 147 145 140 139 138 137 134 131 127 124 120 117 115 112 109 107 106 106 100 96 91 86 86 86 86 87 78 65 52 37 35 41 52 70 81 88 91 86 81 76 71 67 64 59 54 49 45 41 38 36 35 35 36 37 36 36 68 65 58 52 43 37 32 28 28 29 34 39 41 43 47 51 68 89 112 140 154 164 164 158 151 140 136 139 141 145 148 151 153 155 156 156 156 154 150 148 146 144 141 139 137 136 133 130 126 123 119 116 114 111 108 105 104 103 96 89 88 87 90 92 93 93 84 70 57 43 42 50 62 81 95 104 108 103 98 88 79 70 61 55 50 48 47 45 42 40 36 35 34 34 34 34 65 62 56 50 43 38 35 32 31 31 36 42 45 48 52 56 71 90 113 139 153 164 166 162 154 140 134 137 139 143 146 148 149 150 151 150 150 148 145 144 143 142 140 138 136 135 132 129 125 122 118 115 112 109 106 103 102 101 93 87 87 87 91 95 97 97 87 74 62 48 48 56 68 87 101 110 115 110 104 93 82 69 58 52 47 48 49 47 46 42 37 35 33 33 33 33 60 58 53 49 44 40 38 37 35 35 40 46 48 52 57 61 77 95 116 142 156 168 170 167 159 141 132 133 134 135 136 136 137 137 136 135 135 135 135 135 135 136 137 137 135 132 129 126 123 120 116 113 109 105 103 101 100 98 92 85 86 88 92 97 98 99 91 79 68 56 55 61 72 90 102 110 113 109 104 92 80 66 52 48 43 47 51 51 50 45 39 36 34 33 32 32 56 55 51 48 45 43 42 41 39 39 43 49 51 55 60 65 80 98 118 142 155 166 169 167 158 141 131 131 131 131 130 129 129 129 128 127 128 129 130 131 132 133 135 134 133 130 127 124 121 118 114 111 107 103 101 99 98 96 91 86 87 88 91 95 97 99 92 82 73 61 60 64 73 88 98 104 107 103 98 87 75 60 47 44 40 46 51 51 50 44 36 34 33 33 32 32 52 52 50 49 48 47 47 47 46 45 49 53 54 56 62 68 82 99 117 139 151 159 161 158 150 138 130 129 128 127 126 125 125 125 125 126 127 129 131 131 132 132 131 130 129 127 124 121 118 115 111 108 104 101 99 97 95 93 91 89 88 87 89 91 93 95 91 84 77 67 64 65 70 80 86 89 90 88 84 74 64 50 39 37 36 42 47 47 46 39 30 29 31 32 31 31 50 50 49 49 50 50 51 51 51 52 54 57 58 58 64 69 82 98 114 133 142 149 149 146 139 129 122 122 121 120 119 120 120 122 123 125 127 129 130 130 130 129 128 127 126 125 122 119 116 113 109 106 102 99 97 95 93 91 91 92 89 87 87 87 89 92 89 85 79 70 66 64 66 72 75 75 76 74 71 62 53 43 33 34 34 39 44 44 41 34 25 25 29 30 29 29 49 49 50 51 52 53 55 56 58 60 61 62 63 62 66 70 80 92 104 119 125 128 128 124 117 107 102 101 102 103 105 109 113 118 122 126 129 128 126 125 123 123 123 122 121 120 117 115 111 109 105 102 99 97 95 93 91 89 92 94 90 86 84 82 84 87 87 85 80 72 67 62 60 61 61 61 60 56 53 47 41 35 30 32 33 37 42 39 35 29 23 23 25 26 25 25 48 49 51 53 55 56 57 59 62 65 67 67 67 67 69 72 79 86 95 105 109 110 109 105 99 88 83 83 84 87 91 98 105 113 120 125 129 126 123 120 117 117 119 118 117 116 113 111 107 105 102 99 97 95 93 91 89 88 92 96 91 85 82 79 81 84 85 85 80 73 66 59 55 53 50 49 47 43 40 35 32 30 29 33 35 38 41 37 33 28 23 23 24 24 23 23 49 50 53 56 57 58 59 60 64 67 70 72 73 75 75 74 77 79 84 88 90 91 90 87 81 69 63 62 63 67 72 82 92 103 113 119 125 121 118 114 110 111 114 113 112 111 108 106 104 102 99 97 95 93 91 89 88 87 90 94 89 84 81 78 80 82 83 83 78 71 64 56 50 47 43 42 40 34 31 28 27 31 34 39 44 45 47 43 40 35 32 30 29 27 26 26 49 51 55 58 59 60 60 61 65 69 73 76 79 82 80 77 75 73 74 74 74 74 74 72 66 54 48 47 48 51 57 69 81 95 108 115 122 118 115 110 105 106 110 110 108 107 105 103 101 99 97 95 93 91 89 88 87 86 89 92 88 83 80 77 79 81 81 81 77 70 62 53 47 42 38 37 34 28 24 23 24 31 38 45 53 53 53 50 47 43 41 38 34 32 31 31 51 53 57 60 61 62 61 60 64 68 74 79 84 89 85 81 76 70 67 64 64 64 65 65 62 55 50 46 45 47 50 62 74 90 105 114 123 121 119 114 109 107 107 107 105 104 103 101 99 97 95 93 91 90 88 87 86 86 87 89 86 82 80 78 80 81 80 79 75 68 60 51 45 40 37 36 34 26 21 21 23 34 46 54 64 64 63 61 58 57 57 54 49 47 46 46 52 54 59 62 63 63 61 60 64 68 76 83 89 94 89 84 76 67 61 56 55 56 58 60 59 55 51 45 43 44 47 59 70 87 102 113 124 122 121 116 111 108 105 104 102 101 100 99 98 96 93 92 90 89 87 86 86 86 86 86 84 81 80 79 80 81 80 77 73 66 58 50 44 39 36 36 34 26 20 21 22 37 51 62 73 73 72 70 66 68 70 66 61 59 58 58 54 56 62 66 66 65 63 62 72 80 86 91 94 98 93 88 79 70 63 57 54 52 53 54 53 50 48 49 52 58 65 76 87 98 108 113 118 116 113 110 106 103 100 98 97 96 95 94 92 90 88 87 85 84 83 82 81 81 82 83 83 83 82 81 79 76 74 71 67 63 57 51 45 40 35 31 29 32 33 30 28 35 43 59 76 76 77 69 60 60 62 60 57 56 56 56 55 58 65 69 68 67 64 64 79 92 96 99 100 102 97 91 82 72 65 57 53 49 48 49 48 45 46 53 61 72 82 93 102 108 113 113 113 110 106 103 100 98 95 93 91 90 89 88 87 85 84 83 81 80 79 78 77 77 79 80 82 84 83 82 78 72 69 65 62 60 56 51 46 40 34 27 25 38 46 39 33 34 35 57 78 79 81 69 54 53 55 54 53 53 54 54 54 56 59 61 61 60 59 61 76 90 93 96 97 99 95 90 82 73 65 57 53 50 49 50 49 48 50 57 65 77 87 97 105 110 114 112 110 105 99 95 91 89 87 86 85 85 85 84 84 83 83 83 82 81 81 80 80 80 79 77 79 82 81 80 76 70 67 63 61 59 56 51 46 40 35 29 29 40 47 40 34 35 36 56 74 75 76 65 53 52 55 54 53 53 53 53 53 53 53 53 53 53 54 57 73 87 90 93 94 96 92 89 82 74 66 58 54 52 51 51 51 52 55 62 70 83 93 102 109 112 115 110 106 99 92 86 81 79 78 78 79 79 80 80 81 81 82 82 83 83 83 83 83 83 78 74 76 79 78 77 73 67 64 61 59 58 55 50 46 40 36 32 33 42 48 41 36 37 38 54 70 70 70 61 51 51 54 54 53 53 52 52 60 59 56 54 55 55 57 61 73 83 86 89 91 93 90 87 81 74 67 60 56 54 53 52 53 54 57 64 71 83 92 100 106 107 109 102 97 88 80 73 67 66 67 68 69 70 72 73 75 76 78 79 80 81 82 82 83 83 76 71 73 76 75 74 70 64 61 59 57 56 54 50 46 41 38 36 36 43 48 41 37 38 39 52 66 65 64 57 48 50 54 55 55 55 54 54 67 65 59 56 57 58 61 64 72 79 83 86 87 90 87 84 79 73 68 62 58 56 55 54 55 56 59 66 73 82 90 97 102 102 102 94 86 77 66 59 51 52 55 56 58 60 63 65 69 71 74 75 77 78 80 81 82 82 74 67 69 72 71 70 66 61 58 56 55 54 53 50 47 43 41 39 40 45 47 42 38 39 41 50 61 59 57 52 45 48 54 56 57 57 56 56 74 71 65 61 62 63 66 69 72 74 78 81 83 87 85 82 78 73 68 63 59 56 55 54 55 56 58 63 68 75 81 86 89 87 86 79 72 64 55 49 42 44 47 49 51 53 57 59 64 66 69 71 74 75 77 78 79 79 72 66 68 71 70 68 64 60 57 55 55 55 55 52 50 46 43 41 40 42 43 40 38 41 44 51 58 55 52 49 45 48 55 56 55 55 54 54 81 78 72 68 69 70 72 73 71 69 73 76 79 83 82 80 77 74 69 64 61 57 55 54 54 55 56 59 62 67 69 72 73 70 68 62 56 50 43 38 34 36 39 42 44 46 50 53 58 61 64 67 70 72 74 75 76 76 70 64 66 69 68 66 62 58 56 54 55 56 57 55 53 50 46 42 39 39 38 38 38 43 47 52 56 52 48 47 44 49 56 56 53 53 52 52 80 78 73 69 69 69 70 70 69 67 70 73 76 80 80 79 76 73 69 64 61 57 55 53 53 53 54 54 55 57 57 57 57 55 52 48 45 42 39 36 33 36 40 42 44 47 51 54 58 61 64 66 69 70 72 72 73 73 68 64 66 69 68 66 62 58 56 55 56 58 60 60 58 55 51 44 39 36 34 37 40 47 55 58 60 55 51 50 48 53 59 57 53 51 51 51 78 76 73 70 68 67 66 65 65 65 68 71 74 77 78 78 75 72 68 64 61 56 54 51 51 51 50 49 47 45 43 41 38 37 36 35 34 34 35 35 35 39 44 46 48 51 54 57 60 63 66 67 68 69 70 70 69 69 66 64 66 68 67 65 62 58 57 56 58 62 64 65 63 61 56 47 40 34 30 36 42 53 64 66 67 61 56 55 54 58 63 59 52 50 50 50 75 75 72 70 67 64 62 61 62 63 66 69 72 75 76 77 74 72 68 64 60 55 52 49 48 48 47 45 42 38 35 32 29 29 28 30 31 34 37 39 41 46 51 53 55 58 60 63 65 68 69 69 70 70 69 69 67 67 65 64 66 68 66 64 61 58 57 57 59 65 67 69 68 67 62 52 43 35 30 39 48 64 79 79 79 72 65 63 62 64 67 65 59 57 57 57 73 73 71 69 66 62 58 56 59 62 65 68 70 73 74 76 73 71 67 63 59 53 49 46 45 46 45 40 37 32 28 25 22 23 24 28 31 37 42 46 49 55 61 63 65 67 69 71 72 74 74 74 72 71 70 68 66 66 64 63 65 67 65 63 60 57 57 58 61 67 71 73 74 73 69 58 49 37 32 44 57 79 98 97 96 87 77 75 73 73 73 72 70 69 68 68 71 71 70 68 64 60 56 55 59 63 65 67 69 72 73 75 73 71 66 62 57 51 46 43 41 41 40 36 33 29 26 24 21 24 25 30 34 41 47 51 54 60 67 70 72 73 74 76 77 77 77 76 73 71 69 67 65 64 63 62 64 66 64 62 59 56 56 58 62 68 73 76 78 77 73 65 57 43 37 52 68 94 116 114 113 102 91 87 84 82 78 79 79 79 78 78 68 68 68 67 63 59 57 56 60 64 65 66 68 71 72 74 72 70 65 61 56 49 43 39 36 34 32 30 30 28 27 27 26 29 31 36 40 47 52 56 59 65 73 76 77 78 79 80 80 80 78 77 73 70 68 66 63 61 62 61 63 65 63 61 58 55 55 57 62 69 74 79 81 80 78 74 67 51 45 64 83 112 138 136 135 122 109 104 98 92 85 86 90 90 89 89 64 64 64 63 60 58 57 56 60 64 65 66 68 71 72 74 72 70 65 60 54 47 41 36 32 28 26 26 27 27 28 29 30 34 37 41 45 50 55 58 60 67 75 78 79 80 81 82 82 81 79 77 73 69 66 64 61 58 60 60 61 63 62 60 58 55 55 57 62 69 75 80 82 82 81 81 76 59 52 73 94 126 155 153 151 137 123 116 108 100 90 90 94 95 94 94 58 58 57 56 56 56 56 56 61 65 66 67 69 71 72 73 71 69 64 59 52 45 39 33 29 22 19 21 24 26 29 32 36 40 44 47 50 53 56 57 59 66 76 79 80 81 82 82 82 81 78 76 72 68 63 61 58 55 57 59 60 61 61 59 57 54 54 57 62 69 75 81 83 83 84 88 86 67 60 82 105 140 171 168 166 152 135 126 116 107 95 93 94 94 95 95 55 55 54 53 54 55 55 55 59 62 64 65 68 71 72 73 71 69 64 58 52 45 40 35 30 22 18 21 23 28 32 37 42 46 49 51 52 54 56 56 58 66 75 78 81 81 83 84 84 82 79 76 71 67 62 59 57 54 56 58 59 60 59 58 55 53 54 57 63 70 76 81 83 84 85 91 91 76 70 92 114 147 176 175 175 162 147 138 127 118 107 101 97 95 94 94 56 56 55 54 53 54 54 53 52 52 57 61 66 72 73 74 72 68 63 58 54 50 47 44 39 30 25 26 28 32 38 45 51 53 55 53 51 52 52 55 58 65 72 76 80 82 87 88 88 87 83 78 72 68 63 60 58 56 57 58 58 59 58 56 54 52 54 58 64 72 77 81 83 83 84 89 90 87 88 105 122 146 167 173 177 169 161 154 146 140 135 122 104 96 92 89 56 56 55 54 53 53 53 52 48 46 52 57 63 71 72 73 71 68 64 59 56 53 52 51 46 37 31 31 32 36 42 50 57 57 58 54 50 51 51 56 61 67 74 79 82 86 91 92 91 89 84 79 72 67 63 60 58 57 58 58 58 58 57 55 53 51 54 59 65 73 78 81 83 82 83 86 89 94 100 114 129 147 163 172 179 174 169 165 159 157 156 139 115 103 95 92 54 54 53 52 52 52 53 53 50 48 51 53 58 63 66 68 68 68 66 64 61 58 56 55 51 44 40 39 38 41 46 51 56 55 56 53 51 53 56 62 70 79 89 93 95 96 97 96 92 88 81 75 68 63 60 58 56 56 57 57 57 58 57 55 53 51 54 59 65 73 78 81 82 81 81 83 87 94 103 119 134 152 167 174 179 175 170 169 167 168 170 157 137 125 115 111 53 53 52 51 52 52 53 53 51 49 50 50 53 56 60 63 65 67 68 67 66 62 60 58 55 50 47 45 44 46 49 52 55 54 54 52 51 56 62 69 78 89 102 105 105 104 102 98 92 86 78 72 65 60 58 56 55 55 56 56 57 58 57 55 53 51 54 59 65 73 78 81 82 81 80 81 85 94 106 123 139 155 170 174 178 174 170 171 173 176 182 173 158 147 135 130 51 51 50 49 50 51 52 54 52 50 49 47 48 49 53 56 60 65 67 69 69 66 64 61 59 55 53 51 50 50 52 53 54 52 52 53 53 61 70 79 90 100 111 112 110 107 102 96 88 82 73 67 61 57 56 55 56 56 56 56 57 58 57 55 54 52 55 59 65 72 77 80 81 80 79 79 83 95 108 126 143 157 169 171 172 170 167 170 176 182 190 189 184 177 166 162 49 49 48 47 48 50 52 54 52 51 48 45 44 43 47 50 55 62 66 70 71 70 67 64 62 59 57 56 55 54 54 53 53 51 50 53 55 65 77 88 99 108 117 117 113 108 101 94 84 77 68 63 58 55 54 54 56 57 56 56 57 58 57 55 54 52 55 59 65 72 77 80 80 79 78 78 82 96 110 129 146 157 167 167 167 165 163 169 178 187 197 201 205 201 192 188 47 47 46 46 48 50 52 54 51 49 47 44 43 42 44 47 52 57 62 67 68 68 68 66 64 60 58 58 58 57 57 55 53 52 51 54 57 68 80 91 103 106 109 107 103 98 91 85 77 71 65 60 57 55 56 57 59 60 57 55 56 57 56 55 54 53 55 59 64 71 76 79 79 77 77 78 83 97 110 128 143 152 159 159 158 158 157 165 174 184 194 199 202 202 198 196 45 45 44 45 48 50 52 54 51 48 46 44 43 41 42 44 48 53 58 63 65 67 68 68 66 61 59 60 60 60 59 56 53 52 51 55 59 71 84 95 106 104 101 97 93 89 83 77 70 65 61 58 56 55 57 59 61 62 58 55 56 57 56 55 54 53 55 59 64 71 75 78 78 76 76 78 84 97 110 127 141 146 151 151 150 151 152 161 171 181 192 197 200 202 203 203 44 44 43 43 46 48 50 52 49 46 45 45 45 44 44 44 46 48 51 55 58 61 64 66 65 60 59 61 62 63 62 59 56 54 53 57 61 70 79 88 96 94 89 86 83 79 75 71 66 63 61 60 59 59 60 61 62 63 58 54 55 57 57 56 55 53 56 60 65 71 74 76 76 74 75 79 85 96 107 120 131 137 141 143 144 148 151 160 170 179 188 192 193 196 199 201 44 44 43 42 44 46 49 50 47 44 45 46 46 47 46 44 44 43 45 47 51 56 60 65 65 60 59 61 63 65 65 62 59 56 54 58 62 69 75 81 87 83 77 75 73 70 67 65 63 62 61 61 62 62 63 63 63 63 58 54 55 57 57 56 55 53 56 60 65 71 74 75 75 73 74 80 86 95 104 113 122 127 131 135 139 145 151 160 169 177 185 187 186 190 196 199 41 41 40 39 41 42 44 46 44 42 44 46 46 47 46 44 43 40 41 42 45 49 53 58 59 56 56 60 62 65 65 63 60 58 57 61 64 68 71 73 75 75 74 73 71 69 67 66 65 65 65 65 65 65 64 63 62 61 57 54 55 57 57 56 55 53 56 60 65 71 74 75 74 72 72 78 84 92 100 108 115 120 124 131 137 145 154 163 173 179 185 189 193 196 199 200 38 38 37 36 37 37 39 41 40 39 42 45 46 48 47 45 41 37 36 36 39 43 47 52 54 52 53 58 61 65 66 64 62 61 61 64 66 66 66 65 63 66 70 70 68 68 67 67 67 68 69 70 69 68 66 64 61 58 56 54 55 57 57 57 56 54 57 61 65 71 74 75 73 70 70 76 82 89 95 102 109 114 118 126 134 146 158 167 177 182 186 192 199 201 202 202 36 35 34 33 33 33 34 36 37 38 40 42 43 45 44 43 41 37 36 36 37 39 42 44 45 44 45 51 54 59 61 61 61 63 65 69 72 71 70 67 62 65 70 70 68 68 67 67 67 68 70 71 69 68 66 63 59 56 55 54 55 57 57 57 56 54 57 61 65 71 74 75 73 69 69 74 80 88 94 101 107 111 114 123 130 142 155 164 174 179 184 189 196 198 199 199 33 32 30 29 28 28 29 30 34 36 38 39 40 42 41 41 40 38 37 36 36 36 37 36 36 35 36 42 46 51 56 58 60 65 69 75 80 77 74 70 63 65 70 70 68 67 67 67 68 69 71 72 70 68 65 61 57 54 53 53 54 56 56 57 56 54 57 61 65 70 73 74 72 67 67 72 77 86 93 100 105 108 111 119 125 138 151 161 171 176 181 186 191 194 195 195 31 30 27 26 24 24 24 25 31 36 36 36 37 37 38 39 40 40 39 39 38 35 34 31 30 27 27 33 38 44 49 53 58 66 73 81 87 85 82 76 69 68 69 68 66 65 65 65 66 67 69 70 69 68 65 61 57 54 53 53 54 56 56 57 56 55 57 61 65 70 73 74 72 67 67 72 77 87 95 101 106 108 109 115 120 132 144 154 166 171 176 180 186 189 191 192 28 27 24 23 21 20 19 20 29 35 34 33 33 32 35 37 39 42 42 42 40 34 31 26 23 17 17 23 29 36 42 48 56 67 77 87 95 94 92 85 76 72 68 66 64 63 62 63 64 65 67 68 68 67 64 60 57 54 53 53 55 57 57 57 57 56 58 61 65 70 72 73 71 66 66 72 78 89 97 103 108 107 107 111 114 125 136 147 159 164 169 174 180 184 188 190 25 25 24 24 24 25 25 26 31 35 35 36 35 33 34 34 37 41 45 48 46 39 33 24 18 14 14 19 25 34 42 54 66 80 93 101 109 104 98 88 77 71 65 64 63 62 62 63 64 65 66 66 65 64 62 60 58 56 55 55 56 57 57 57 57 57 59 62 66 70 72 73 72 68 68 74 79 87 93 99 104 105 107 111 115 124 134 144 154 159 164 170 179 183 186 187 22 23 26 28 31 33 35 37 36 35 39 43 40 37 33 29 33 39 47 56 54 47 37 22 14 12 13 17 22 34 45 65 82 101 117 122 126 115 103 90 76 69 61 60 61 63 64 65 65 64 64 63 61 60 60 59 59 58 58 58 58 56 56 56 58 59 61 64 68 71 72 73 73 71 72 76 80 83 86 92 97 102 106 113 118 126 134 142 150 154 159 168 179 183 184 184 24 25 29 32 36 39 41 43 39 37 42 47 45 41 37 32 35 40 47 55 53 47 36 20 13 12 15 24 34 52 68 88 105 118 129 128 126 112 98 85 72 65 59 58 60 62 63 64 65 64 63 62 59 58 57 56 56 56 57 59 58 56 56 55 57 59 61 64 67 70 72 73 73 72 74 80 84 85 86 91 94 100 106 114 121 128 136 142 150 153 157 165 174 175 170 169 30 31 34 37 41 44 46 47 43 41 46 51 49 48 43 39 39 41 45 49 47 40 32 19 14 13 19 39 58 84 106 124 137 139 138 127 115 101 85 75 65 60 56 56 58 60 62 63 65 64 63 61 58 56 53 52 50 50 54 59 58 56 55 54 56 58 60 63 66 69 71 72 73 73 76 86 91 90 90 92 93 99 105 115 124 132 140 145 151 154 156 161 166 161 148 143 32 33 36 39 43 46 48 49 45 42 47 52 51 50 46 42 41 42 44 46 44 38 30 19 14 13 21 45 67 97 121 138 150 147 142 127 111 96 80 71 62 58 55 55 57 59 62 63 65 64 63 61 58 55 52 50 48 48 53 59 58 56 55 54 55 57 59 63 66 69 71 72 73 73 77 88 94 92 91 92 93 99 105 115 125 133 141 146 151 154 156 159 163 155 139 133
## 1862                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         173 172 172 172 171 172 173 173 173 174 174 174 174 174 174 174 174 174 174 174 172 175 162 88 44 45 29 29 42 31 10 15 27 23 22 30 42 61 89 112 132 135 141 144 148 151 151 151 151 149 134 115 96 81 59 52 53 42 33 23 17 17 11 8 4 13 25 28 21 7 3 4 3 3 6 7 16 22 6 1 2 1 0 0 1 3 3 3 3 3 3 2 7 9 6 2 173 172 172 172 171 172 173 173 173 174 174 174 174 174 174 174 174 174 174 173 171 163 153 97 54 49 36 23 28 31 26 27 27 33 50 65 78 95 126 145 149 152 151 151 153 157 158 159 159 159 144 125 105 88 64 47 46 36 32 26 20 15 7 12 17 12 20 19 19 15 9 11 9 8 7 7 18 25 17 11 11 7 1 0 0 1 0 0 0 0 0 1 5 6 4 1 173 172 172 172 171 172 173 173 173 174 174 174 174 174 174 174 174 174 174 175 177 169 162 119 67 52 47 40 34 41 47 50 46 52 75 103 127 138 150 153 150 154 154 154 156 157 158 159 160 158 148 136 121 101 74 55 48 34 28 19 15 14 14 22 19 22 26 9 10 17 11 13 18 21 23 20 19 21 17 11 10 9 4 2 1 0 1 1 1 1 0 1 3 3 3 1 173 172 172 172 171 172 173 173 173 174 174 174 174 174 175 175 175 175 175 172 177 179 172 132 66 55 58 50 51 62 66 70 81 96 116 141 156 156 156 156 160 162 160 159 160 161 162 161 157 154 143 139 130 112 83 61 53 39 34 25 17 15 12 14 9 9 14 10 10 9 7 10 14 17 18 12 5 4 2 3 3 3 3 4 2 0 0 0 0 0 1 3 1 0 1 3 173 172 172 172 171 172 173 173 173 174 174 174 174 174 175 175 175 175 175 174 170 177 172 137 76 63 58 56 64 78 97 112 125 142 153 162 165 160 162 160 161 163 162 162 163 162 161 156 151 149 143 138 130 114 88 63 56 47 45 35 34 34 29 26 24 13 12 19 17 11 12 11 8 4 2 1 3 3 2 4 3 1 1 0 0 2 2 2 2 2 3 4 1 0 1 4 172 172 172 172 173 173 174 174 174 173 173 173 173 173 173 173 173 173 173 175 174 182 177 160 98 72 77 101 104 114 137 150 154 164 163 163 162 161 163 161 160 163 161 160 161 163 160 154 150 149 144 137 130 116 90 68 54 50 56 50 48 47 36 32 30 20 16 19 13 9 13 6 12 11 8 4 4 3 3 3 2 0 3 2 2 4 2 2 3 4 3 1 0 0 0 1 172 172 172 172 173 173 174 174 175 173 173 173 173 173 175 175 175 175 175 175 174 176 176 157 91 79 117 151 156 157 171 176 172 172 166 164 162 161 161 160 160 162 161 161 163 166 158 149 148 151 150 148 132 118 96 72 57 45 55 49 47 46 40 42 42 33 23 14 5 5 8 14 30 19 13 9 6 5 4 3 2 3 4 2 0 1 1 1 1 2 2 0 0 0 0 0 172 172 172 172 173 174 174 175 175 175 175 175 175 175 177 177 177 177 177 178 178 179 179 145 88 106 161 179 181 184 187 183 173 170 168 166 163 161 161 161 161 161 161 162 164 165 158 150 148 152 153 152 126 118 113 86 69 47 45 54 53 48 50 53 48 33 23 16 10 10 8 10 16 7 8 8 6 5 3 3 3 4 3 2 0 0 0 0 1 1 1 1 1 1 1 1 173 173 173 173 173 173 174 174 175 176 176 176 176 176 176 176 176 176 176 176 177 178 175 138 114 154 189 187 188 192 187 178 171 172 170 169 165 163 162 162 162 162 163 163 161 158 157 151 148 148 151 155 134 117 109 90 83 70 47 51 50 44 49 52 52 41 27 11 8 12 12 9 8 9 13 6 1 1 1 3 4 1 0 2 3 2 1 1 0 0 0 1 1 1 1 1 173 173 173 173 173 174 174 175 175 176 176 176 176 176 175 175 175 175 175 177 173 172 173 156 161 196 199 197 196 194 189 180 177 173 171 170 167 165 165 166 163 163 165 163 157 153 154 150 147 148 151 153 145 120 106 102 85 69 56 51 48 39 37 40 52 51 40 24 16 11 2 4 7 5 9 2 1 0 1 1 2 1 0 2 3 1 2 1 1 0 0 1 1 1 1 1 174 174 175 174 174 175 175 176 176 176 176 176 176 176 176 176 176 176 176 179 176 181 172 173 193 200 204 202 195 192 187 183 180 175 172 171 168 167 166 165 164 164 163 160 156 152 151 150 149 150 149 148 139 129 109 103 95 77 72 54 56 50 37 30 42 46 44 39 29 19 21 28 30 15 17 30 16 9 8 5 11 11 5 1 2 3 0 2 0 0 0 1 2 2 1 0 174 174 175 176 175 175 176 176 176 177 177 177 177 177 177 177 177 177 177 177 175 177 173 190 204 201 205 201 193 190 185 182 179 174 170 168 166 164 164 164 166 165 163 161 158 152 150 149 150 152 148 138 132 136 114 97 112 106 90 68 71 66 56 42 36 42 48 45 31 19 18 27 40 41 42 47 31 26 31 22 20 26 16 4 6 3 1 2 2 1 1 1 0 0 1 3 174 175 176 176 175 175 176 176 177 178 178 178 178 178 177 177 177 177 177 175 180 176 181 197 205 206 202 197 193 190 186 185 183 176 171 168 166 164 164 164 165 163 161 159 158 156 152 149 150 152 147 133 124 128 120 106 102 110 110 91 70 64 68 55 38 36 44 57 55 37 13 3 10 35 47 33 22 24 27 14 9 18 10 3 2 1 3 0 3 4 2 1 3 4 2 2 175 176 175 176 176 176 177 177 177 178 178 178 178 178 177 177 177 177 177 178 180 174 188 204 204 203 198 194 193 187 181 181 179 173 170 171 168 167 166 165 161 159 157 155 156 160 155 152 149 149 146 137 122 115 119 114 99 105 109 93 77 67 58 52 52 42 31 39 51 42 25 20 19 32 49 51 54 47 31 16 5 2 12 20 6 4 3 1 1 2 3 7 3 1 2 5 176 175 175 176 176 176 177 177 178 177 177 177 177 177 177 177 177 177 177 177 175 176 197 205 200 202 195 192 188 183 180 176 175 172 171 172 169 167 167 166 161 159 156 155 156 157 157 156 153 152 148 139 128 117 112 109 114 113 105 90 77 79 56 48 50 50 43 32 32 38 31 20 14 23 43 56 63 54 41 30 9 1 13 22 14 9 8 7 3 4 7 7 6 3 2 2 175 176 177 177 177 177 176 176 176 177 177 177 177 177 176 176 177 177 178 179 173 184 201 200 199 199 192 187 185 181 179 176 176 174 171 170 167 166 164 163 161 159 157 157 156 155 158 159 156 154 152 144 130 118 116 117 116 110 110 97 74 72 62 43 44 52 53 52 41 28 29 26 17 21 32 41 48 46 35 31 18 14 5 7 12 7 2 0 0 0 0 0 4 8 4 2 176 178 177 177 177 177 176 176 175 177 177 177 177 176 176 177 177 177 178 179 176 191 203 198 196 194 188 183 182 181 177 174 173 171 169 169 166 164 163 161 157 157 157 159 159 158 161 159 155 151 155 151 135 119 115 130 123 104 99 93 80 62 58 35 37 50 47 58 69 53 28 21 22 22 28 41 53 52 39 32 32 29 15 19 14 5 15 32 55 60 45 26 5 0 3 7 177 177 178 178 177 177 177 177 176 179 178 178 177 176 177 177 177 178 178 177 181 197 203 195 192 191 185 181 179 179 175 172 171 168 166 166 164 162 161 159 156 156 156 158 159 159 160 159 155 152 150 146 139 128 120 113 114 104 95 82 80 67 48 40 39 42 46 42 53 64 53 34 26 24 30 44 51 59 62 41 36 30 20 24 26 59 109 130 138 133 125 117 76 26 0 1 178 178 178 177 177 177 178 178 177 179 179 179 179 179 177 178 178 179 179 179 185 199 198 194 192 189 186 181 180 178 172 169 169 167 166 164 163 161 161 160 159 157 156 155 156 156 157 157 155 154 147 137 133 132 128 113 100 93 98 92 76 71 59 44 54 56 60 57 47 51 51 42 45 39 27 32 34 45 68 55 42 32 4 27 98 141 149 130 117 113 109 123 142 118 54 9 178 178 177 178 177 177 178 178 179 180 180 181 182 183 179 179 179 180 180 178 188 201 194 190 190 186 185 181 177 174 169 166 167 167 166 164 162 161 162 162 161 159 157 156 156 156 156 154 153 152 147 135 136 135 123 119 105 93 91 90 77 58 68 58 53 64 67 58 58 62 55 49 43 33 30 34 35 42 48 51 36 12 18 101 151 122 76 55 42 43 50 64 114 148 125 71 179 179 177 178 178 177 179 180 180 179 182 184 184 184 180 180 180 180 180 178 190 202 194 189 188 185 184 178 176 172 170 169 169 168 166 164 163 162 162 162 159 160 158 158 158 158 158 154 150 148 144 139 143 141 129 121 120 116 102 92 76 49 57 73 66 59 68 58 63 70 66 62 38 29 29 39 46 46 41 36 17 12 74 153 111 46 26 50 56 69 95 84 95 125 138 118 179 179 178 178 178 178 180 180 179 180 183 186 185 183 181 182 182 182 182 181 191 201 194 190 188 185 184 177 176 171 170 168 169 166 163 163 162 162 162 161 159 159 159 157 157 159 160 155 149 146 148 145 145 147 149 139 132 130 127 117 99 76 61 58 75 61 63 57 52 60 68 67 42 37 33 43 46 36 37 28 13 53 131 133 53 32 74 112 117 116 126 120 115 122 136 134 180 179 179 178 177 180 180 179 178 182 186 188 187 184 181 182 183 184 185 181 191 200 195 191 189 185 184 177 176 171 168 166 166 161 160 162 162 160 159 158 157 157 157 156 154 153 157 154 150 151 149 144 144 149 154 148 140 140 141 134 120 104 83 62 69 63 65 59 45 56 62 64 49 47 44 50 46 30 30 35 60 98 137 104 27 44 114 144 134 126 123 121 123 121 135 142 180 180 179 179 178 180 181 180 180 186 188 190 189 186 183 183 183 184 184 182 191 200 195 191 189 185 184 177 176 171 170 168 168 162 163 165 163 160 155 153 153 153 153 152 151 151 155 151 146 147 147 144 144 147 148 145 144 139 136 137 126 111 98 79 70 71 67 60 54 54 48 54 49 48 49 53 49 34 29 62 115 112 111 87 22 41 111 144 152 151 126 117 119 125 141 145 181 180 180 180 179 180 180 183 186 186 189 192 191 188 188 188 188 187 187 184 192 201 195 192 189 185 184 176 176 171 170 169 169 166 163 165 165 160 155 152 149 149 148 149 148 150 155 147 139 143 147 144 146 148 147 142 142 140 137 138 127 109 102 88 73 71 66 62 65 56 51 55 46 33 42 48 43 50 49 91 124 105 114 94 17 33 98 133 159 170 143 119 108 112 127 142 181 180 180 181 181 180 181 185 188 188 191 194 193 192 194 193 192 191 189 185 193 201 195 195 191 184 183 180 178 172 170 169 168 166 164 163 163 164 162 155 146 147 146 146 146 146 148 145 142 145 145 142 142 145 146 140 136 138 140 135 124 114 110 94 75 64 63 63 69 57 59 65 41 21 34 37 36 66 85 111 120 105 117 96 27 53 112 122 137 162 157 124 114 113 113 140 181 179 181 183 182 183 183 185 186 191 194 196 198 198 197 195 194 193 191 189 196 200 195 197 193 186 183 184 180 174 170 168 165 161 165 168 165 168 165 156 145 141 140 141 144 144 141 141 143 144 145 145 145 143 140 140 137 139 141 131 124 116 111 103 82 64 62 70 75 56 57 65 38 25 29 27 32 69 104 115 117 110 119 100 51 72 121 122 132 148 148 124 112 110 106 135 182 180 181 184 186 186 186 187 188 188 194 200 205 205 201 196 196 198 194 196 200 199 196 196 191 186 183 178 177 175 171 166 165 164 166 167 161 159 151 144 139 129 128 131 134 137 134 133 138 143 142 146 146 142 137 138 137 139 139 130 125 117 112 104 83 67 64 76 72 48 57 60 42 36 35 36 49 81 107 116 112 106 121 99 73 106 117 119 144 154 151 132 109 97 101 131 184 182 183 186 189 190 191 192 191 185 192 205 212 212 204 200 199 200 199 202 201 198 196 194 187 186 184 181 180 175 167 162 163 165 165 163 154 145 134 130 134 130 130 128 126 129 127 126 132 142 143 144 144 141 138 137 133 132 132 128 124 118 110 96 74 69 63 69 63 42 59 57 41 42 37 54 87 107 109 117 119 116 125 93 83 146 153 158 167 156 154 159 125 94 91 118 187 185 186 189 190 192 194 196 195 192 198 212 218 212 203 202 199 197 199 202 198 196 196 190 185 185 185 183 176 168 159 153 153 153 149 145 135 127 121 121 129 130 126 123 122 121 123 121 127 137 138 139 138 138 137 135 132 129 128 124 124 123 113 92 73 70 60 61 64 52 61 56 54 64 64 82 114 123 118 129 125 122 131 102 51 66 111 133 136 137 149 171 146 101 82 103 187 186 187 190 190 192 195 199 199 201 207 218 219 213 204 203 199 193 198 200 194 195 194 185 177 174 174 170 159 146 136 129 127 121 113 111 109 110 115 118 118 117 110 112 121 124 122 119 125 133 131 132 134 134 133 134 133 132 129 125 124 126 117 97 83 89 83 83 93 88 92 90 91 96 93 104 125 129 127 130 118 118 123 114 55 9 53 63 66 108 135 153 150 111 82 102 187 185 186 189 190 190 192 198 202 205 213 219 217 217 209 205 198 193 193 197 191 195 194 180 169 160 153 145 131 117 107 100 96 87 78 79 87 93 98 102 103 105 101 107 120 124 115 114 123 129 126 125 129 131 131 134 134 133 130 128 125 127 122 112 109 119 117 112 115 115 114 114 108 104 104 112 127 135 131 116 100 100 108 115 82 35 102 118 94 107 109 125 148 128 86 98 192 190 190 191 192 193 190 193 201 211 215 216 217 219 213 206 199 192 192 195 191 195 190 173 158 145 133 122 104 92 84 76 70 64 67 75 79 79 79 80 85 92 90 87 93 105 107 115 123 124 120 120 126 130 130 135 134 133 131 132 130 132 129 125 126 128 123 116 112 111 114 113 108 105 110 120 130 131 125 112 100 102 95 106 109 65 89 124 133 138 110 91 128 140 91 93 200 200 201 201 199 200 195 195 204 210 212 211 213 216 215 209 204 200 198 194 190 192 181 160 147 132 115 101 80 74 68 62 61 60 68 76 77 78 80 84 84 85 88 94 102 107 108 111 116 115 116 117 125 130 131 136 135 134 134 133 133 132 127 124 127 122 120 115 108 109 113 113 106 105 113 121 127 125 122 120 116 116 90 90 115 89 83 101 108 128 130 102 120 140 96 95 204 206 208 207 206 207 203 202 205 206 209 209 210 214 214 211 209 207 202 198 189 184 164 139 126 113 94 75 58 56 54 56 61 65 76 79 81 80 82 86 87 88 91 91 90 94 92 98 107 111 117 119 127 132 134 137 138 138 136 132 132 131 130 128 128 125 124 119 111 108 112 112 108 108 114 119 123 126 128 130 132 134 100 81 98 104 102 111 114 125 150 151 146 134 95 99 212 213 213 212 209 208 208 205 202 205 208 210 211 214 214 211 211 209 199 194 184 168 141 112 96 84 70 55 45 41 36 46 56 67 78 76 81 82 82 87 89 83 80 77 72 70 80 94 108 116 119 122 130 135 136 136 140 140 136 131 132 131 129 127 123 121 120 116 112 109 108 109 106 105 106 115 123 122 130 143 147 147 138 108 85 92 98 111 125 131 152 160 162 125 85 104 218 213 211 212 210 207 204 202 202 203 206 210 214 215 210 210 215 215 194 192 186 164 128 94 79 66 57 46 32 26 30 60 78 83 73 51 46 45 49 53 64 72 69 66 72 84 92 101 110 120 131 133 140 144 141 135 136 136 133 135 135 132 129 125 122 120 119 116 110 109 110 110 104 101 102 110 123 125 134 146 153 156 164 140 74 65 97 103 111 116 124 139 150 109 79 112 222 217 216 214 211 208 204 201 201 202 207 212 213 212 209 214 214 218 185 174 190 164 122 94 81 66 55 44 31 28 35 61 65 46 28 10 14 22 15 14 31 49 56 72 96 103 97 106 129 145 149 144 143 144 144 142 137 131 131 137 136 135 133 129 125 120 117 115 112 110 108 106 100 97 101 111 121 127 141 148 158 160 170 149 78 46 81 98 103 117 128 143 133 96 90 118 229 225 222 219 214 210 205 202 201 204 209 214 213 211 212 209 210 222 170 139 189 173 128 103 88 73 62 56 48 42 45 48 29 6 9 47 84 109 94 77 79 84 85 89 96 104 119 129 139 148 148 146 142 140 138 143 139 132 130 135 135 135 134 129 124 114 110 110 112 110 107 106 102 97 103 113 119 128 144 147 156 162 164 132 79 38 66 115 127 140 148 145 114 86 111 136 232 230 227 225 218 213 209 207 209 209 213 215 215 214 215 212 216 222 158 100 153 186 154 110 93 76 62 58 50 48 58 57 33 5 30 103 111 136 122 116 122 121 129 125 116 127 143 143 138 143 146 149 148 144 143 139 139 135 130 130 130 130 127 122 117 110 104 107 111 110 108 110 105 99 105 110 112 125 140 141 154 169 163 99 43 37 96 157 158 148 145 124 101 100 130 134 234 232 230 228 222 217 213 214 215 214 215 217 217 218 210 210 214 219 178 81 94 179 174 130 106 78 67 65 60 60 68 68 57 25 47 102 73 117 129 136 127 121 130 138 136 136 143 143 143 147 147 150 148 145 145 137 138 135 130 129 127 123 118 114 109 108 104 106 113 112 109 111 109 100 102 108 114 130 145 152 166 177 167 80 20 68 146 175 150 126 133 124 107 121 139 122 236 233 231 230 227 222 220 221 219 217 220 221 220 219 216 216 218 221 208 101 45 124 178 161 133 103 82 73 76 81 82 74 64 59 56 74 108 147 153 136 126 132 147 153 145 135 139 142 144 145 147 150 150 152 147 138 133 129 128 126 125 119 112 109 106 101 98 101 109 110 110 118 116 103 102 105 114 131 152 165 165 163 151 104 89 142 174 163 134 114 131 130 109 124 134 111 238 237 237 236 231 229 228 228 225 224 226 226 224 222 220 217 219 213 217 176 81 48 150 174 143 136 116 101 100 101 104 91 85 85 63 93 148 155 144 126 124 137 146 149 149 148 146 147 147 149 153 155 155 155 147 137 132 126 125 121 119 113 109 107 106 100 94 96 102 103 108 120 120 108 105 106 109 127 144 145 143 135 125 133 159 167 138 132 122 111 141 121 111 126 128 98 239 237 237 236 233 232 232 232 230 228 230 230 227 224 219 216 215 212 210 215 196 81 69 163 169 154 147 135 127 130 133 118 127 110 85 141 158 141 112 108 129 151 156 149 147 146 147 147 148 149 152 154 155 155 145 132 127 121 119 115 108 106 107 107 101 105 100 99 101 103 110 120 124 113 104 103 103 122 136 129 126 121 103 115 143 132 105 111 120 114 130 111 118 135 117 75 239 236 235 234 235 233 234 233 229 228 229 230 228 225 223 221 216 221 207 205 216 189 87 147 193 166 158 147 139 153 152 139 145 134 117 159 153 105 71 101 140 159 164 158 153 144 148 151 153 153 157 158 159 159 146 130 124 118 115 112 104 105 110 110 102 110 105 103 104 108 125 131 124 112 110 105 101 115 129 125 115 110 103 107 127 131 120 117 127 125 121 124 138 136 92 44 239 238 238 238 237 236 236 233 230 227 229 230 230 228 224 218 214 217 209 203 198 211 183 173 182 180 160 142 152 169 150 137 142 132 128 118 99 90 118 154 163 163 160 159 162 159 159 160 162 163 162 160 159 156 139 130 126 120 118 114 112 112 114 115 113 108 98 100 104 114 128 131 126 112 103 100 102 108 120 123 111 101 104 123 142 156 143 132 150 146 135 143 134 100 56 34 239 238 238 238 237 236 235 233 229 230 230 230 228 227 224 219 215 211 209 205 199 197 198 187 184 179 161 151 164 170 150 138 138 130 125 119 125 144 167 172 165 166 165 163 161 157 159 159 160 165 163 159 158 151 137 128 124 119 116 114 113 112 113 111 110 108 104 107 107 114 124 126 123 110 98 97 101 102 118 121 117 101 110 139 149 159 144 132 149 146 138 141 111 70 44 36 239 238 238 238 237 236 235 234 230 231 230 228 227 224 221 217 213 209 208 203 197 201 198 189 188 177 154 149 161 163 161 155 150 146 145 159 171 175 178 177 172 173 170 167 163 161 163 161 158 158 157 151 153 149 134 127 123 118 114 112 112 112 109 107 102 109 112 112 110 112 120 121 114 105 99 98 103 105 109 115 120 101 110 155 163 159 150 140 145 135 126 115 77 45 33 36 239 239 238 238 238 237 235 234 230 230 229 225 222 222 217 214 210 207 206 203 197 200 197 187 183 175 153 154 162 157 169 167 158 162 161 167 176 181 186 183 176 177 173 170 167 163 164 161 153 148 145 138 145 147 132 123 120 115 110 108 111 112 109 106 103 110 114 115 116 114 117 118 114 109 104 104 108 106 104 112 111 97 106 146 166 155 150 145 136 112 98 84 44 19 26 38 240 239 239 239 238 237 237 235 231 230 227 222 219 220 211 209 207 205 205 204 199 200 191 180 178 173 158 160 164 159 172 169 161 172 167 165 176 186 180 170 169 171 171 169 166 162 159 154 148 144 139 133 140 141 129 121 118 115 112 111 113 113 111 111 111 111 114 116 120 119 117 119 120 118 111 105 102 99 104 116 103 100 106 123 154 156 140 130 115 96 79 54 30 19 28 37 240 240 239 239 239 237 236 235 231 230 226 221 218 218 207 205 204 205 206 205 204 203 189 178 182 173 161 158 160 166 175 174 172 177 176 179 185 184 181 172 161 165 165 162 160 158 150 146 145 144 141 136 137 134 123 121 119 119 119 120 116 114 112 112 112 112 115 115 120 120 115 116 119 119 111 102 99 97 96 108 99 99 95 93 109 130 123 106 92 69 42 25 26 28 31 34 239 239 238 238 238 238 238 236 233 230 225 221 217 213 206 202 206 210 208 206 207 201 188 177 179 179 168 164 170 172 179 176 171 179 183 185 186 183 178 171 161 162 162 158 154 153 147 142 142 138 137 133 133 129 122 118 121 124 123 121 114 112 111 113 116 115 116 116 118 122 117 113 114 113 107 102 99 97 95 95 95 93 84 79 78 79 71 60 49 30 13 24 28 31 28 30 239 238 238 237 237 237 238 235 234 230 225 223 217 210 208 205 208 213 211 209 210 199 182 172 173 179 179 179 176 170 172 174 179 184 182 183 183 180 174 168 162 161 161 156 150 147 142 137 138 135 132 128 126 122 118 115 119 123 120 118 116 115 115 116 116 112 119 120 114 117 117 113 113 110 108 104 99 97 95 93 89 87 88 85 75 61 41 28 22 23 23 28 31 30 26 34 238 237 237 237 236 236 236 235 233 230 227 225 219 211 212 210 212 215 214 215 212 194 176 170 167 178 186 179 170 165 165 171 181 183 181 183 182 177 171 165 159 157 155 151 145 138 133 130 134 132 128 124 119 116 115 116 120 122 121 120 120 121 121 118 115 113 117 118 114 113 115 114 110 108 110 106 98 95 93 90 86 87 84 79 70 63 56 42 33 29 26 27 33 31 32 48 237 236 236 235 235 236 235 234 231 229 227 226 221 214 215 215 217 217 217 220 209 182 173 171 163 175 182 164 155 161 164 169 175 182 185 181 178 172 167 162 155 150 147 142 136 130 127 127 133 131 125 120 115 113 117 123 125 125 126 126 123 123 122 120 117 119 114 113 119 115 112 112 110 109 108 105 96 92 93 88 84 83 80 75 66 59 63 67 66 56 45 41 45 48 56 75 236 236 235 235 235 235 235 232 231 229 226 225 222 218 217 218 219 218 218 219 200 169 169 171 163 167 172 158 145 146 143 154 172 178 178 175 171 165 161 157 151 145 141 134 127 126 128 130 135 131 124 118 113 114 122 128 128 128 128 127 126 125 124 125 125 123 119 119 120 114 106 108 113 110 105 101 93 89 92 89 81 78 77 74 64 59 68 74 79 81 81 82 75 81 90 100 234 235 235 234 234 233 233 232 231 229 227 225 222 222 217 218 216 217 220 214 188 157 164 169 164 163 159 159 145 131 122 133 160 174 170 171 168 160 153 151 147 139 132 128 124 123 128 128 131 128 124 124 117 116 126 130 130 129 127 127 128 128 128 126 130 123 121 124 117 108 108 112 110 111 102 99 96 91 93 88 80 81 76 71 65 65 71 73 76 86 101 105 98 102 109 110 233 232 233 233 232 232 231 231 231 228 227 225 222 221 217 218 215 217 220 207 178 157 162 168 166 158 153 157 151 136 121 117 130 163 172 167 165 159 149 142 141 133 126 124 121 117 119 121 124 121 124 127 123 124 133 136 136 136 133 128 135 138 132 130 128 126 122 116 115 115 112 110 107 109 101 100 100 95 92 87 80 80 76 72 70 69 71 76 79 85 95 100 98 101 106 107 231 231 230 230 229 229 229 228 228 227 226 223 221 219 218 219 216 216 218 199 171 161 158 164 160 146 147 150 149 152 151 139 124 140 163 160 151 147 142 134 134 129 123 120 118 114 114 120 127 125 124 124 126 132 137 137 135 137 136 127 129 132 129 134 127 131 122 112 120 111 104 106 108 107 100 100 102 97 89 86 82 78 76 73 72 70 72 80 82 84 89 91 96 100 104 106 230 230 228 228 228 227 227 226 226 226 224 222 220 219 216 217 214 213 214 192 170 164 152 156 148 135 141 148 149 143 144 144 126 122 148 159 145 136 138 133 130 127 123 118 116 112 114 125 130 129 125 125 126 130 130 131 128 130 131 127 119 122 132 141 135 127 125 114 111 110 104 103 112 108 101 100 102 97 87 84 83 79 76 72 70 69 71 78 81 85 91 90 96 102 106 108 230 229 227 227 227 226 226 225 225 225 224 221 219 219 212 213 211 207 208 185 171 166 150 153 146 136 136 151 156 144 137 133 120 102 122 148 147 134 137 134 126 124 120 115 114 114 120 130 128 128 127 128 127 127 129 129 130 131 129 123 123 126 143 143 135 128 130 115 107 114 109 101 111 103 99 100 99 93 86 85 82 77 74 70 68 68 71 79 81 84 91 94 94 102 108 108 229 228 226 225 224 224 225 224 223 222 222 219 218 216 212 213 208 211 207 178 169 162 154 156 145 137 144 162 165 156 144 132 120 88 97 131 146 137 130 130 124 123 119 115 115 122 129 132 127 127 127 126 125 126 129 130 134 134 130 124 130 133 143 134 127 133 127 118 114 107 103 98 101 95 95 96 96 92 86 85 79 75 72 72 70 67 73 79 81 83 88 93 94 102 109 107 228 227 225 224 222 222 223 221 221 220 219 218 216 213 212 212 205 211 205 173 162 161 168 165 147 145 158 168 165 155 136 121 109 84 97 127 138 136 129 128 130 125 122 123 126 129 134 128 123 126 124 124 123 124 123 126 131 129 128 131 133 135 134 129 130 133 121 116 110 105 102 92 92 89 94 92 90 85 79 79 77 74 71 74 71 66 74 81 81 84 90 95 94 100 108 108 227 226 223 223 224 224 223 222 221 220 218 215 212 210 209 209 203 209 200 171 153 150 159 165 150 148 154 155 142 127 108 87 70 64 111 147 135 133 138 133 131 127 126 130 135 137 139 128 120 124 121 121 122 123 121 122 125 123 121 123 122 133 134 134 132 126 122 115 109 108 106 96 94 89 92 88 82 77 74 76 76 71 70 71 70 67 73 82 81 84 89 95 93 98 107 108 227 225 223 222 219 218 215 215 215 215 214 212 209 208 207 209 208 214 198 169 153 143 137 139 141 132 124 118 99 73 44 33 50 82 132 158 139 132 142 140 136 135 136 138 138 139 141 132 121 121 121 117 119 123 125 124 120 123 127 117 113 128 132 131 129 123 124 121 117 113 102 90 94 91 91 84 78 80 80 76 73 70 69 69 69 67 71 79 81 83 89 90 94 105 112 109 225 224 222 220 213 212 208 208 208 211 211 210 208 206 206 207 205 214 202 169 151 141 125 115 107 97 89 57 23 13 13 39 84 113 119 138 148 138 136 142 142 142 143 141 141 139 138 131 125 124 124 119 121 125 124 125 124 125 125 119 111 121 130 127 122 123 125 121 121 117 101 86 89 90 92 84 79 85 84 76 74 72 70 67 66 67 70 79 83 88 93 92 98 110 115 111 225 223 220 218 214 213 210 209 208 208 208 207 206 205 207 203 201 210 204 169 141 132 118 98 81 66 49 15 9 28 74 116 114 97 87 111 145 149 144 147 148 145 143 141 141 143 133 129 129 129 126 122 124 124 120 122 124 117 111 114 112 121 131 123 119 128 125 119 120 113 102 91 89 88 87 83 81 85 83 77 75 73 69 66 64 66 72 81 85 91 97 98 103 111 115 110 222 220 217 214 212 210 208 206 206 207 206 205 204 203 204 204 204 209 206 176 132 115 107 82 73 54 23 10 36 70 106 112 83 67 75 85 107 136 150 154 159 153 151 147 143 143 130 126 125 127 129 125 123 121 119 123 127 123 115 110 116 121 115 111 118 127 118 113 114 101 92 89 87 88 85 80 81 84 83 78 74 70 67 64 64 64 71 79 83 87 93 98 103 109 111 107 220 218 214 212 209 206 204 204 206 205 205 205 203 202 205 203 199 204 209 196 145 94 69 63 58 35 26 55 68 79 85 71 75 71 75 78 85 106 135 153 149 150 150 147 141 134 127 124 124 126 130 129 125 123 124 126 126 124 121 117 115 114 113 114 117 116 108 104 104 96 91 88 87 90 85 79 80 81 81 78 75 70 65 62 63 65 70 76 79 81 87 95 104 107 104 104 221 218 215 212 207 205 202 203 204 202 203 203 201 199 201 201 195 199 206 205 193 154 96 61 51 72 115 167 130 73 78 79 87 84 80 74 81 92 114 136 137 142 147 144 139 134 129 125 126 129 131 133 129 125 126 124 117 112 115 120 120 114 113 116 121 112 104 102 102 99 94 86 86 89 84 79 79 78 78 77 77 75 64 58 61 65 69 75 75 78 82 88 102 108 108 110 219 216 214 211 209 208 204 203 203 203 204 205 202 199 198 202 201 203 201 195 206 210 196 179 179 195 193 164 106 74 102 99 78 78 77 66 74 85 93 112 127 133 143 144 140 136 133 126 129 135 136 135 132 125 118 118 122 119 112 107 114 113 109 108 109 110 108 103 98 96 96 88 83 88 84 78 78 76 76 76 76 68 61 56 58 62 68 75 76 79 84 90 105 114 115 116 216 214 211 209 209 208 206 205 203 206 207 208 206 202 203 204 202 204 203 197 194 193 197 199 192 159 118 90 87 93 108 100 78 71 60 55 60 60 63 79 91 104 119 127 133 130 131 129 135 142 134 135 138 130 121 119 124 122 111 108 104 108 112 110 104 111 111 103 96 90 95 90 82 84 82 77 78 76 74 74 69 60 58 56 57 60 67 75 77 81 88 102 111 115 113 111 214 213 211 210 208 206 206 206 202 204 205 205 204 202 203 204 204 203 204 203 197 194 191 179 156 116 90 91 87 89 96 89 83 78 62 57 55 52 55 58 68 88 99 111 135 135 135 134 137 137 134 138 143 136 130 129 123 116 108 116 113 105 105 111 107 109 108 102 99 90 88 86 80 77 78 76 78 75 71 67 61 56 54 53 57 62 70 75 77 81 89 103 107 108 109 111 212 212 210 209 207 206 205 205 202 203 203 203 202 201 203 204 204 206 206 209 205 200 186 173 155 124 106 103 108 108 95 76 77 80 67 63 62 57 55 51 51 69 73 85 113 129 132 133 137 138 133 131 135 132 126 125 123 118 111 115 117 105 107 112 101 98 100 96 91 84 82 80 75 76 76 75 75 70 67 62 52 49 52 55 60 65 74 76 76 81 93 106 106 105 108 112 210 211 210 209 209 207 206 206 202 202 201 201 200 200 204 203 203 206 208 208 205 200 184 175 170 131 110 118 134 126 108 87 78 77 70 66 66 64 59 55 55 59 63 65 79 110 127 136 140 136 129 129 132 131 122 115 115 113 106 109 112 100 105 110 101 95 94 94 91 88 82 77 77 78 77 72 66 62 63 55 47 49 56 61 63 67 76 77 78 85 97 105 107 108 109 111 210 211 211 210 211 208 208 208 203 201 200 198 199 199 204 203 203 204 207 206 203 201 184 177 175 137 120 122 128 115 105 93 82 78 71 70 74 74 67 60 57 60 63 52 51 87 117 134 136 127 125 127 126 125 117 110 104 99 97 108 111 100 105 107 107 101 98 96 91 91 88 82 78 76 76 69 60 59 56 47 43 48 59 65 66 67 76 79 81 90 100 104 110 112 114 115 210 211 211 212 212 210 209 209 205 202 200 198 199 200 204 202 201 202 205 205 203 205 188 177 168 142 134 118 97 82 77 74 74 74 73 70 71 74 63 55 58 68 61 38 31 60 95 120 127 119 118 116 110 110 108 103 100 97 95 103 105 108 118 112 107 107 103 95 89 89 88 82 75 73 73 68 58 57 50 40 40 46 58 67 68 68 76 80 84 93 102 107 111 114 116 119 210 209 209 211 211 211 210 208 207 203 200 198 199 201 199 200 200 201 201 200 200 206 199 190 187 158 136 119 59 39 53 43 44 59 70 57 51 64 62 59 66 70 62 42 37 47 70 109 125 116 114 109 100 96 99 103 103 104 106 109 111 114 115 112 109 104 95 86 83 83 80 77 73 70 67 63 52 47 43 38 46 54 62 68 69 70 78 82 87 94 103 112 112 113 114 116 209 208 208 208 209 211 211 211 209 203 200 198 199 200 199 199 199 200 200 200 199 201 200 199 200 185 166 166 128 73 72 43 22 19 24 56 70 83 83 75 80 64 59 53 51 47 50 87 111 114 114 106 94 91 94 99 103 107 112 114 115 115 113 109 104 97 92 88 85 82 79 75 73 69 62 54 45 42 42 42 53 60 63 66 67 69 78 82 87 96 103 106 109 113 117 120 208 206 205 205 208 209 211 212 211 204 201 198 198 199 198 198 198 197 197 197 196 196 197 201 202 200 197 196 197 151 117 74 21 11 48 105 119 106 95 84 87 73 64 62 62 65 52 59 94 112 106 101 92 90 95 99 103 111 120 121 118 117 112 106 101 98 95 92 90 85 78 72 68 62 56 50 45 43 47 52 60 64 64 65 66 69 76 81 88 97 104 108 110 114 115 116 206 204 204 204 206 208 211 211 211 204 201 198 198 199 198 197 197 197 196 195 197 195 198 200 195 199 210 201 205 191 160 110 63 87 141 153 121 92 87 90 91 83 78 78 72 75 63 49 80 101 101 97 94 93 96 98 99 112 124 128 120 116 112 106 104 99 95 90 84 76 74 70 63 57 52 47 45 46 51 58 61 59 60 62 64 68 74 80 87 96 105 108 110 114 115 114 205 203 203 203 206 207 209 209 209 205 202 198 198 198 197 197 196 196 195 195 199 198 202 203 201 198 204 205 200 195 190 138 102 147 161 124 86 82 86 100 107 88 87 91 74 65 60 56 72 89 101 102 103 106 111 111 108 115 125 127 120 116 112 107 108 103 94 89 80 72 69 66 62 58 49 42 43 45 49 53 52 52 53 56 59 64 74 79 86 94 103 107 110 114 115 114 203 201 200 201 205 206 207 207 206 203 202 200 197 195 196 197 197 198 198 198 199 201 203 203 202 201 200 200 200 193 199 171 142 165 133 86 79 85 95 115 125 99 90 94 70 61 60 63 67 81 94 97 101 107 115 118 113 115 122 123 120 120 115 107 105 100 94 86 79 71 65 60 56 52 44 40 40 41 48 49 44 45 50 49 51 56 66 72 78 90 97 108 112 115 111 112 202 200 199 200 202 205 207 205 203 201 200 198 197 195 196 197 197 198 198 198 197 199 200 201 199 198 198 198 200 194 195 190 184 153 94 71 73 79 106 131 135 110 105 96 69 66 61 63 63 69 84 87 95 102 106 108 107 110 117 121 124 125 117 106 100 96 89 82 76 69 63 56 46 38 36 34 33 33 36 38 37 35 41 41 47 52 59 66 73 85 93 99 115 113 114 115 202 199 199 199 200 203 206 203 199 198 198 196 196 194 196 197 197 198 198 197 197 198 198 199 198 197 197 199 201 203 195 189 189 114 49 47 51 74 116 135 132 116 107 97 79 67 55 54 57 57 72 78 93 103 104 106 108 113 118 121 127 124 113 104 102 92 82 76 69 61 56 50 38 29 27 25 23 24 22 24 27 33 40 46 55 64 70 76 81 91 95 106 112 110 112 111 202 200 199 199 200 203 204 201 199 198 198 196 196 195 196 197 197 198 198 197 196 198 197 197 199 199 199 200 203 202 195 192 187 127 81 68 57 80 111 130 131 122 101 88 87 75 59 53 56 55 63 64 74 90 100 111 115 116 113 113 119 115 102 94 98 91 81 73 65 58 50 39 29 24 19 16 18 22 25 30 37 47 56 62 72 81 89 92 93 101 103 110 102 106 104 124 203 201 200 200 202 201 202 200 200 197 197 197 196 196 196 197 197 198 198 198 197 197 198 197 198 198 196 197 200 198 197 196 190 179 164 132 98 87 89 114 105 91 102 98 83 79 65 53 50 46 56 62 69 80 90 104 111 112 109 108 109 104 92 84 87 83 74 65 57 48 42 29 17 18 16 12 18 29 41 50 56 64 73 77 83 89 95 97 97 103 107 110 106 113 108 166 205 205 204 203 200 202 202 199 197 195 195 195 195 195 195 195 195 195 195 197 197 197 197 197 197 198 196 197 202 203 193 198 190 186 178 162 130 85 82 100 81 75 94 106 87 81 73 62 48 40 46 58 67 76 85 100 106 109 107 104 99 96 91 85 75 74 70 60 52 42 35 27 18 15 19 19 24 40 54 61 63 65 73 78 82 88 90 94 96 102 107 111 115 107 130 197 206 205 206 203 201 201 200 198 197 195 194 194 195 194 194 194 194 194 194 196 196 196 196 196 196 197 196 197 201 202 193 197 195 200 187 155 113 81 84 96 89 97 100 101 98 87 78 69 55 44 46 44 57 76 85 93 98 103 102 98 97 96 93 84 71 63 62 56 43 30 24 24 22 16 16 19 29 44 52 58 61 62 69 75 77 85 89 90 93 99 109 112 110 105 165 203 206 207 205 203 202 201 199 198 196 195 195 194 194 194 194 194 194 194 194 195 195 195 195 195 195 195 195 196 199 200 197 200 198 204 184 144 99 85 83 88 102 110 100 87 90 86 75 74 67 54 49 39 47 64 77 86 87 95 99 100 100 96 90 80 71 60 58 52 36 27 23 23 20 12 13 18 27 36 37 40 46 50 60 68 72 80 87 90 90 93 99 104 95 123 201 191 206 204 204 202 201 199 197 195 194 195 193 194 194 193 194 194 194 194 194 194 194 194 194 194 195 194 194 198 199 199 200 200 196 191 160 135 116 90 81 85 93 99 91 75 81 72 62 65 68 56 42 36 38 44 61 77 84 93 96 100 92 82 72 67 64 59 53 41 27 21 23 24 19 13 16 17 16 17 17 19 25 29 41 50 51 56 66 75 79 83 92 90 84 164 212 168 204 203 203 202 201 198 195 194 194 193 193 193 193 193 194 194 194 194 194 194 194 194 194 194 194 192 193 198 198 199 198 195 196 195 144 106 111 92 88 79 80 89 80 64 82 76 63 63 68 60 45 39 42 39 49 70 87 96 90 88 82 70 61 58 60 50 40 29 24 16 16 22 25 24 18 12 6 3 6 12 19 25 36 39 29 31 49 59 63 69 84 76 103 202 193 165 202 201 201 200 198 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 194 194 195 196 197 196 195 198 176 118 97 112 93 88 76 79 81 74 67 85 74 62 60 63 65 60 48 50 43 34 51 75 87 84 80 79 71 63 58 54 40 36 27 17 27 36 28 30 30 15 13 15 8 4 18 36 44 46 40 32 34 45 43 44 53 63 60 148 223 173 171 200 200 199 199 197 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 195 194 196 196 199 198 198 200 164 111 113 111 97 84 81 82 80 77 79 94 74 62 53 49 49 47 40 41 53 40 36 52 61 75 74 69 71 64 50 42 35 27 23 26 35 40 27 11 3 5 12 17 22 16 29 55 62 59 50 52 60 56 50 48 46 35 78 202 207 165 182 198 197 197 197 197 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 195 194 195 196 196 194 197 199 180 139 120 101 106 103 88 81 86 88 81 89 86 68 51 50 49 49 46 43 54 46 41 42 43 56 68 65 63 56 45 41 32 21 32 45 27 10 13 15 20 14 5 2 21 32 34 45 52 51 45 53 66 68 67 59 52 43 129 222 183 169 198 198 197 197 197 197 197 195 196 195 195 195 195 194 194 195 195 195 195 195 195 195 195 195 195 195 195 194 195 196 194 195 198 197 197 181 141 108 114 120 97 88 82 83 73 73 75 69 60 58 54 49 49 49 42 43 45 40 40 36 51 58 55 51 41 29 24 34 36 24 40 78 116 152 162 145 114 73 43 17 19 35 45 43 41 45 52 61 61 63 55 70 185 210 170 172 207
## 6493                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            128 128 128 128 128 128 128 128 128 129 133 133 133 132 132 132 132 132 132 133 133 133 133 133 133 132 132 131 131 136 140 142 142 142 142 142 142 143 143 143 143 143 142 143 143 143 143 143 143 142 142 141 139 139 144 149 148 148 148 148 148 148 148 148 148 148 148 149 155 154 152 152 155 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 161 159 158 164 168 167 128 128 128 128 128 128 128 128 128 129 133 133 133 132 132 132 132 132 132 132 132 132 132 132 133 132 132 132 136 141 144 142 142 142 142 142 142 143 143 143 143 143 143 142 142 142 142 142 142 142 142 143 144 144 146 148 148 148 148 148 148 147 147 147 147 147 147 148 155 154 152 152 155 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 164 164 164 166 167 167 128 128 128 128 128 128 128 129 129 130 132 132 132 132 133 133 132 133 132 133 133 134 135 134 135 132 135 137 140 143 144 142 142 142 142 142 142 143 143 143 143 143 143 143 143 143 143 143 144 145 145 146 148 148 148 148 148 148 148 148 149 150 150 150 150 150 150 152 156 155 154 154 156 157 157 157 157 157 157 158 158 158 158 158 159 160 160 162 163 163 165 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 133 132 136 138 135 138 133 140 143 142 142 142 142 142 142 142 142 142 143 143 143 143 143 143 143 143 143 143 142 145 148 148 148 147 147 147 148 148 148 148 148 150 153 153 153 153 153 156 158 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 132 131 135 137 134 137 132 139 142 142 142 142 142 142 142 142 142 142 143 143 143 143 143 143 143 143 142 143 143 145 148 148 148 148 148 148 148 148 148 148 148 150 152 152 152 152 152 155 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 132 131 133 136 138 136 138 137 140 142 142 142 142 143 143 143 143 143 143 143 143 143 143 143 145 144 144 145 145 145 146 148 148 148 148 148 147 148 148 148 148 147 150 155 157 156 155 155 156 157 157 157 157 157 157 157 157 157 157 157 157 158 158 159 160 160 161 163 163 163 163 163 165 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 131 128 134 137 137 137 139 143 142 142 142 142 142 143 143 143 143 143 143 143 143 143 143 142 146 147 146 150 148 148 148 147 147 147 147 147 147 148 148 148 148 148 146 147 153 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 163 163 163 163 163 165 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 131 128 134 137 137 137 139 142 142 142 142 142 142 143 143 143 143 143 142 142 142 142 143 142 147 146 145 149 147 147 148 147 147 147 147 147 147 147 147 147 147 148 146 145 151 155 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 163 162 162 162 162 166 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 132 130 133 136 137 137 139 142 142 142 142 142 142 143 143 143 143 143 144 145 145 145 145 145 148 147 147 149 148 148 148 147 147 147 147 147 147 148 148 148 148 148 150 153 157 158 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 161 161 164 164 164 164 166 167 167 167 167 167 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 132 132 134 137 137 139 142 142 142 142 142 142 142 142 142 142 142 145 148 148 148 148 148 147 148 148 147 148 148 148 147 147 147 147 147 148 148 147 148 148 147 149 152 152 154 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 160 160 166 168 168 168 168 168 168 168 168 168 128 128 128 128 128 128 129 132 132 132 132 132 132 133 133 133 133 133 133 132 132 134 137 137 139 142 142 142 142 142 142 143 143 143 143 143 145 148 148 148 148 148 147 148 148 148 148 148 148 147 147 147 147 147 148 147 147 147 147 147 149 152 152 154 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 160 159 165 167 167 167 167 167 167 167 167 167 128 130 130 130 130 130 130 132 132 132 132 132 132 132 133 132 131 133 135 134 134 136 139 139 140 142 142 142 142 142 142 143 143 143 143 143 145 148 147 147 148 148 147 148 148 148 148 148 148 147 147 147 147 147 147 148 148 148 148 151 153 154 154 155 157 157 157 157 157 157 157 157 157 158 158 158 158 158 159 160 160 162 163 163 162 162 166 167 167 167 168 168 168 168 168 168 129 132 132 132 132 132 132 132 132 132 132 132 132 133 133 131 129 135 137 137 137 139 142 142 142 143 143 143 143 143 142 143 143 143 143 143 144 148 148 148 148 148 148 147 147 147 147 147 147 148 148 148 148 148 148 147 147 147 149 156 158 157 157 157 157 157 158 158 158 158 158 158 157 158 158 158 158 158 161 163 163 163 163 163 165 168 167 167 167 167 167 167 167 167 167 167 129 132 132 132 132 132 132 132 132 132 132 132 132 132 133 130 128 134 138 136 137 139 142 142 142 143 143 143 143 143 143 142 142 142 142 142 144 148 148 148 148 148 148 147 147 147 147 147 147 148 148 148 148 148 148 147 147 146 148 155 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 167 167 167 167 130 132 132 132 132 132 132 133 132 133 132 133 133 133 132 132 131 135 136 140 139 140 142 142 142 143 143 143 143 143 143 143 143 144 145 145 146 147 147 147 147 147 147 147 147 147 147 147 147 148 148 148 148 148 149 149 149 150 152 156 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 167 168 169 169 132 132 132 132 132 132 132 132 132 132 132 132 134 132 130 134 134 135 133 143 142 142 142 142 142 143 143 143 143 143 143 143 142 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 147 148 148 148 148 148 150 152 153 155 158 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 167 170 172 172 133 132 132 132 132 132 132 133 133 133 133 133 135 131 128 133 132 133 133 143 142 142 142 142 142 143 143 143 143 143 143 143 143 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 147 148 148 148 148 148 150 152 151 153 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 158 158 158 161 163 163 163 163 163 165 167 167 167 167 167 167 167 166 170 172 172 131 132 132 132 132 132 132 133 133 133 133 133 134 136 134 137 137 138 137 143 142 142 142 142 142 143 143 143 143 143 143 143 146 147 147 147 147 147 147 147 147 147 147 147 147 148 150 150 150 150 150 149 148 148 150 153 156 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 159 160 160 161 162 162 164 165 165 166 167 167 167 167 167 167 169 171 172 172 172 129 132 132 132 132 132 132 133 133 133 133 133 134 142 140 141 143 144 143 143 143 143 143 143 142 143 143 143 143 143 143 145 150 151 148 148 148 147 147 147 147 147 148 148 147 150 153 153 153 153 152 151 149 149 150 154 161 160 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 163 163 166 168 168 167 167 167 167 167 167 168 171 177 175 173 173 129 132 132 132 132 132 132 132 132 132 132 132 133 141 139 141 142 143 143 142 142 142 142 142 142 143 143 142 143 143 143 145 150 151 148 148 148 147 147 147 147 147 148 148 147 150 153 153 153 151 152 153 154 154 153 154 160 160 157 157 157 157 157 157 157 157 157 158 158 158 158 158 158 158 158 161 163 163 163 162 162 165 168 167 167 167 167 167 167 167 168 171 176 175 173 173 130 132 132 132 132 132 132 133 133 134 135 135 137 143 136 138 143 143 143 143 143 143 143 143 143 143 143 144 145 145 145 145 148 149 147 147 147 147 147 147 147 147 148 148 147 149 151 150 152 155 154 155 157 157 156 155 159 159 157 157 157 157 157 157 157 157 157 158 158 158 158 158 159 160 160 162 163 163 163 165 165 166 167 167 168 167 167 168 169 169 170 172 175 174 173 173 132 132 132 132 132 132 132 133 132 134 138 138 137 135 138 141 142 142 142 143 143 143 143 143 143 142 142 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 148 148 148 147 147 147 152 158 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 163 163 163 163 163 165 168 168 168 168 168 167 167 167 170 172 172 173 173 173 173 173 173 132 132 132 132 132 132 132 132 132 133 137 137 136 135 139 142 144 144 143 143 143 143 143 142 142 142 142 144 148 147 148 147 147 147 147 147 147 147 147 147 147 147 147 148 148 147 147 146 152 158 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 157 158 158 158 158 158 161 162 163 163 163 163 165 167 167 167 167 167 167 167 167 170 172 172 173 173 173 173 173 173 132 133 133 133 133 133 133 133 132 135 136 138 139 141 136 136 136 139 138 140 140 140 143 149 142 144 144 145 148 148 148 147 147 147 147 147 147 147 147 147 147 147 148 150 150 150 151 151 153 156 156 156 158 158 157 155 156 157 159 159 158 157 157 157 157 157 156 157 157 157 157 157 162 166 165 165 165 165 166 167 167 167 167 167 169 170 170 171 172 172 173 173 173 173 173 173 132 132 132 132 132 132 133 132 131 136 134 139 136 137 137 135 134 132 149 156 157 154 148 145 148 156 155 154 152 152 153 154 153 154 154 153 154 154 153 155 156 156 155 153 152 156 162 161 158 154 154 157 160 160 158 156 156 155 155 154 160 166 166 166 166 166 166 166 166 167 166 166 167 168 168 168 168 167 169 169 169 169 169 169 172 173 173 174 175 175 174 173 173 174 175 175 132 132 132 132 132 132 133 132 131 136 134 139 136 138 138 136 137 144 106 77 74 79 128 133 136 135 135 136 139 139 143 148 147 148 148 147 148 148 146 150 156 155 156 153 153 154 156 155 157 158 158 160 163 163 164 164 165 163 162 161 164 167 165 164 163 162 161 159 158 158 158 160 163 165 167 169 169 168 173 177 176 176 176 176 175 172 172 177 180 180 175 171 171 176 180 179 132 133 132 133 133 132 132 132 131 134 133 139 139 135 135 136 131 147 41 39 77 77 122 132 129 127 124 124 122 119 117 124 122 125 126 124 121 119 122 126 130 130 123 128 126 129 132 133 132 132 132 134 135 135 138 142 140 143 146 145 147 154 164 169 174 184 191 198 202 205 204 193 186 180 170 163 162 162 164 167 166 168 169 168 170 171 170 177 182 181 178 176 176 181 185 184 132 133 133 133 133 133 133 132 132 132 134 139 138 132 133 130 124 129 44 103 172 155 149 148 150 151 150 148 147 148 151 153 151 145 154 165 163 161 159 161 152 159 158 163 170 159 166 155 161 154 155 154 154 154 153 151 152 150 148 148 148 151 158 160 164 171 171 185 205 213 207 189 176 169 163 155 150 151 150 150 150 144 141 141 142 142 142 143 144 144 145 146 145 147 149 148 132 133 133 133 133 133 133 132 131 135 141 136 133 132 133 130 124 130 41 9 20 35 60 80 84 114 152 186 211 214 175 90 150 164 140 89 89 90 97 95 65 72 66 51 52 62 69 59 109 168 159 162 161 161 160 158 158 160 162 161 162 162 159 159 161 157 156 156 151 152 153 149 156 165 167 166 163 164 164 164 164 166 167 167 165 166 165 167 169 169 165 162 163 162 160 160 133 132 132 132 132 132 132 132 132 134 138 134 131 132 133 132 132 133 91 33 61 91 88 119 122 135 175 213 222 217 122 14 103 125 88 96 98 81 110 99 102 66 38 15 25 87 65 78 100 166 154 154 157 157 156 159 159 158 157 158 157 157 161 160 159 162 164 164 164 164 165 167 167 165 162 161 162 161 161 162 161 165 168 168 165 163 163 168 172 171 174 177 177 175 173 174 133 132 132 132 132 132 132 132 132 132 131 132 133 133 133 132 132 129 124 50 66 96 99 133 150 158 177 202 186 141 44 12 55 53 65 113 108 97 116 105 113 92 90 55 70 107 118 90 120 168 156 154 160 160 154 156 157 157 157 157 160 163 163 163 164 163 163 163 163 163 163 163 162 162 162 163 163 163 162 163 162 165 167 166 169 170 170 171 172 172 172 172 172 174 175 175 132 132 132 132 132 132 132 132 132 132 132 132 132 132 132 131 131 131 111 63 84 111 136 164 170 169 166 151 113 49 18 12 63 85 103 120 103 106 111 120 116 117 95 59 96 118 113 84 117 169 157 154 157 162 161 160 159 160 161 159 161 164 163 162 161 161 161 161 160 161 161 161 162 161 162 162 161 162 162 162 162 166 170 168 173 179 178 176 176 175 174 174 174 178 181 181 132 132 132 132 132 132 132 132 132 132 132 132 132 132 132 132 129 128 104 106 146 164 173 179 151 101 76 53 93 90 42 23 89 72 77 113 98 77 97 76 93 87 72 77 61 85 102 117 122 160 154 162 169 149 124 138 140 138 138 151 152 155 162 166 172 173 170 171 174 171 173 169 172 176 168 166 168 165 162 166 162 164 159 161 169 157 161 162 154 166 166 162 161 173 177 175 132 132 132 132 132 132 132 132 132 132 132 132 133 133 133 134 127 127 100 170 163 141 122 110 88 32 52 56 49 76 85 84 64 64 87 108 59 84 93 72 98 112 116 116 114 95 71 65 102 155 137 149 139 59 36 62 79 81 96 73 84 87 87 98 126 136 155 167 173 166 165 167 173 180 168 162 164 159 147 145 141 136 105 97 92 76 98 99 75 98 100 104 78 124 194 167 132 132 132 132 132 132 132 132 132 132 132 132 133 133 133 134 127 125 107 174 139 98 90 108 120 74 88 114 97 126 126 132 117 74 81 46 31 80 88 115 123 94 95 105 110 115 119 99 111 156 149 114 30 44 32 52 75 69 84 53 58 61 59 67 92 101 129 135 136 154 158 150 151 158 155 150 147 144 124 117 114 113 84 108 106 71 95 134 99 113 116 112 96 88 152 172 132 133 133 133 133 133 133 132 132 132 132 132 133 132 132 132 131 124 118 165 124 96 101 95 78 83 97 103 92 71 75 89 54 36 10 63 66 39 23 108 106 52 76 93 101 79 101 118 122 158 164 123 55 81 92 118 114 67 49 52 62 81 98 103 111 106 120 121 130 161 163 152 136 145 145 137 132 132 122 113 108 109 86 108 118 85 90 127 93 115 114 98 99 93 86 168 133 133 133 133 133 133 133 132 132 132 132 132 133 133 133 133 136 127 118 156 124 130 88 57 52 65 46 63 54 108 116 108 102 69 23 28 28 67 75 78 78 93 132 110 89 53 90 73 94 163 163 125 73 133 133 126 109 67 66 56 81 89 108 128 128 121 124 124 151 170 157 163 140 131 138 156 155 139 131 127 124 119 101 117 104 97 122 122 69 96 112 97 93 91 98 175 132 133 133 133 133 133 133 132 132 132 132 132 132 133 133 133 136 126 120 151 94 73 35 82 129 137 97 26 14 58 63 44 124 87 85 61 99 123 108 96 73 89 53 49 34 40 74 44 68 163 173 122 64 123 121 111 95 67 79 68 83 83 102 129 136 128 135 124 128 151 148 150 121 121 148 167 169 142 125 134 139 134 127 144 122 108 122 120 69 74 103 86 84 79 89 174 132 133 133 133 133 133 134 135 135 135 135 135 134 132 132 132 135 126 115 155 95 48 45 115 135 126 130 51 15 10 26 68 71 87 120 122 114 100 107 91 16 35 33 24 70 106 102 53 54 163 170 121 65 101 98 110 95 63 68 65 94 120 136 140 147 142 138 138 158 161 140 154 145 143 155 153 168 158 132 131 140 145 142 145 133 124 132 115 82 71 81 72 68 69 80 173 132 133 133 133 133 133 134 137 137 137 137 137 136 132 132 132 135 125 120 137 73 37 64 96 107 108 81 60 16 17 44 66 70 118 118 68 65 114 96 93 31 32 35 39 100 121 139 91 46 165 164 122 76 99 88 99 102 86 79 90 113 130 144 150 155 154 170 184 175 173 162 162 177 172 167 155 158 158 139 139 153 155 142 138 131 126 139 110 103 103 111 81 89 66 81 172 132 133 132 133 133 132 134 137 137 137 136 137 136 133 133 133 136 126 119 128 86 40 90 122 125 119 108 99 29 56 44 62 113 97 67 43 90 77 22 55 57 18 68 99 54 99 98 58 55 162 165 129 86 102 87 102 115 88 86 108 126 146 159 167 162 171 183 174 173 172 170 160 162 171 176 174 167 163 155 162 153 144 142 139 133 123 125 121 124 117 90 92 117 83 89 173 132 133 137 136 133 139 136 139 136 138 140 138 133 135 131 130 134 126 124 109 74 52 90 133 101 91 130 117 66 127 138 90 17 26 107 133 122 108 39 13 31 43 103 131 121 132 107 99 66 163 164 128 93 101 84 111 103 95 137 143 133 153 169 171 165 184 191 181 178 169 167 168 164 165 169 171 163 189 193 192 185 163 145 138 136 124 110 98 128 162 115 84 68 81 90 176 132 133 140 138 133 144 137 141 136 139 142 140 132 136 130 127 132 126 129 116 67 41 89 120 91 85 117 106 71 121 73 37 8 90 133 131 124 144 79 20 18 56 120 123 123 117 129 108 62 167 159 130 106 114 84 79 98 138 169 168 143 157 172 166 162 185 187 183 179 174 170 171 171 171 170 175 164 180 186 187 196 194 174 151 131 117 104 85 127 149 150 123 74 78 119 171 132 133 139 138 133 143 136 140 135 135 133 142 138 135 130 127 132 126 130 111 72 76 99 115 117 123 117 54 24 94 101 38 34 59 70 117 84 99 106 13 48 116 114 130 106 76 127 59 70 171 149 134 113 119 92 88 125 162 163 177 163 166 173 163 157 180 182 177 172 172 175 175 175 175 176 175 168 164 182 184 187 193 195 169 140 129 117 103 135 159 154 144 96 75 115 171 132 134 139 139 135 140 138 142 140 140 140 142 139 135 130 127 132 127 128 125 121 132 128 108 81 93 85 27 14 46 40 13 95 116 110 121 86 95 114 20 32 119 102 107 125 118 75 35 98 162 153 129 96 105 89 87 97 139 159 168 166 167 174 165 153 176 181 174 168 175 183 182 181 182 176 177 173 151 176 175 187 180 191 179 150 128 109 96 106 142 148 135 102 93 95 177 132 134 139 140 137 137 139 143 143 145 147 141 138 135 130 127 132 126 128 129 104 65 104 142 107 106 110 33 18 53 22 16 87 136 120 116 134 127 126 62 60 128 116 63 73 96 42 76 67 158 159 127 84 98 86 64 83 145 160 167 177 179 177 155 146 179 180 177 172 169 168 168 176 167 163 177 186 180 186 177 168 190 197 188 161 132 111 98 89 147 153 163 149 111 98 172 132 133 139 140 136 137 139 143 143 145 147 141 138 135 130 127 132 127 130 130 93 73 107 120 126 132 89 25 19 54 28 20 71 130 93 89 116 128 99 49 110 120 119 127 50 73 122 104 50 163 158 126 80 92 87 103 131 166 163 173 188 196 194 176 150 175 189 183 177 173 170 171 172 164 170 177 165 181 176 172 178 190 192 182 154 134 121 109 124 167 167 163 167 153 111 167 134 139 141 142 140 140 141 143 143 144 145 142 141 138 133 130 131 125 131 136 135 125 106 116 135 131 88 35 24 9 17 25 40 123 111 101 128 131 47 23 94 95 91 128 119 62 88 65 59 154 157 125 78 85 132 158 151 158 168 176 183 192 202 202 173 171 184 186 194 174 173 173 164 179 190 170 150 170 173 169 187 171 164 165 157 138 125 121 169 183 169 162 161 166 138 167 135 144 143 143 143 143 142 143 143 143 143 143 143 140 135 132 131 124 131 137 131 121 119 122 131 129 122 84 88 71 26 13 17 63 107 119 104 56 24 36 74 133 90 105 133 69 90 119 77 146 155 125 54 97 167 148 149 158 165 175 177 188 200 196 188 177 183 179 190 196 186 181 185 189 173 146 147 152 153 145 144 150 151 157 160 139 125 149 174 173 169 161 163 157 139 168 135 143 143 143 142 142 142 143 143 143 143 143 143 140 135 132 131 124 126 143 128 121 119 128 125 130 136 100 88 131 74 44 81 90 82 89 102 39 16 21 75 134 113 69 134 103 60 120 104 143 156 124 34 116 164 147 151 157 160 169 176 190 201 195 196 184 186 174 172 200 203 189 181 182 170 149 146 145 147 143 144 147 147 153 151 134 127 168 175 174 164 161 162 158 141 168 138 141 140 140 143 143 143 143 143 143 143 143 143 140 135 133 133 125 134 147 120 114 123 125 119 130 130 86 51 61 81 116 103 121 132 126 133 97 37 39 96 127 123 97 123 117 48 102 101 146 153 125 38 131 155 156 144 148 154 165 173 186 195 196 203 194 195 188 189 183 187 189 184 193 187 163 152 160 157 149 150 150 152 156 152 139 135 169 180 179 169 166 163 161 122 165 137 136 138 140 143 142 143 143 143 143 143 143 143 140 135 133 134 125 145 144 125 112 98 115 115 119 130 125 81 55 124 124 121 125 121 121 125 126 121 122 118 128 120 124 121 121 36 84 111 149 152 131 45 131 154 156 151 144 149 158 168 185 193 196 207 200 203 198 200 195 187 181 181 197 199 178 155 157 157 154 154 150 153 151 155 150 143 175 181 174 171 166 162 156 102 171 135 130 141 147 143 143 143 143 143 143 143 143 143 140 135 133 134 123 150 137 112 116 111 99 105 116 133 129 120 126 119 124 126 119 123 123 124 118 121 121 112 130 120 113 110 127 48 81 129 150 156 129 64 129 156 153 147 154 151 153 152 159 166 177 214 204 198 197 197 196 193 188 186 198 196 183 163 156 158 161 161 158 160 161 153 141 134 173 182 177 170 161 156 154 104 169 134 134 139 142 140 140 140 143 143 143 143 144 141 139 139 134 137 127 161 135 97 83 77 77 70 124 128 122 122 126 125 130 132 125 122 123 120 115 124 102 110 114 102 106 119 135 56 96 124 148 154 120 83 147 154 134 94 121 140 140 117 122 137 168 209 198 192 196 199 199 192 187 183 195 201 183 178 164 173 181 175 164 161 164 161 154 155 174 172 160 151 151 152 159 121 166 134 138 137 137 137 137 139 143 143 143 143 144 140 137 141 134 135 125 191 149 78 54 45 25 96 119 117 128 126 125 125 123 123 123 121 121 122 127 132 98 92 109 118 121 119 126 87 120 115 147 151 119 92 163 143 100 64 76 76 75 80 97 112 145 195 194 178 181 187 194 189 185 178 184 197 176 175 156 163 165 160 159 166 181 196 184 169 170 165 150 145 149 152 155 119 171 133 137 137 137 138 138 139 143 143 143 143 144 140 138 141 135 133 120 216 156 79 67 64 63 123 106 115 126 125 128 125 121 121 122 125 126 128 125 119 104 115 113 119 112 116 120 126 127 119 148 151 122 71 143 141 82 70 77 68 67 78 86 93 122 165 182 184 177 184 192 191 186 184 180 194 184 174 145 166 171 174 187 192 207 205 191 192 188 178 176 165 157 152 153 98 175 135 141 141 139 134 134 137 143 143 143 143 143 142 140 138 130 133 113 210 162 78 60 59 108 125 117 119 108 97 89 90 98 99 103 102 95 90 105 93 79 87 88 106 122 119 118 123 117 118 146 149 130 15 67 148 54 62 58 59 64 74 97 85 125 154 170 185 186 189 194 188 177 184 182 191 185 175 148 163 172 178 203 198 199 199 194 190 182 174 172 167 158 156 151 85 172 137 143 143 140 135 135 137 143 143 143 143 143 142 141 136 130 133 114 191 154 91 67 70 114 123 111 118 118 109 118 111 108 113 114 112 107 99 98 94 110 110 105 111 117 114 120 117 118 116 146 151 122 26 57 132 61 91 89 65 54 77 97 112 131 151 163 167 177 181 190 185 174 183 182 187 179 171 152 167 188 187 208 201 203 199 195 190 181 174 168 164 161 162 136 80 172 141 143 143 142 141 141 141 143 143 143 143 143 142 141 136 133 134 123 182 163 105 80 113 115 126 110 111 116 112 100 119 108 112 108 113 121 126 112 107 111 109 119 124 119 118 120 119 118 117 147 149 130 29 46 41 41 96 96 92 94 96 99 74 92 122 133 143 157 170 183 180 180 184 176 183 175 174 161 175 200 189 203 203 203 200 194 193 185 176 172 171 169 143 72 53 182 142 139 140 139 139 139 141 142 142 142 142 142 143 141 136 134 134 125 158 156 107 94 120 114 124 94 91 100 99 46 46 79 67 45 48 77 95 101 97 87 89 93 104 102 100 119 114 115 114 142 150 122 14 29 63 29 45 69 57 71 85 93 105 124 133 143 141 147 159 174 185 185 178 167 173 175 175 174 180 191 183 189 196 197 195 194 170 158 157 152 141 132 102 51 57 184 141 137 136 139 146 146 145 143 143 143 143 143 143 141 136 134 135 126 141 149 145 135 113 103 95 56 79 115 90 72 58 83 114 105 83 103 103 104 106 103 100 99 88 37 86 117 111 112 112 139 149 119 10 47 56 24 53 64 83 78 52 52 67 75 96 116 115 119 138 148 166 186 172 162 171 173 192 200 186 181 168 142 168 191 184 176 129 105 102 98 79 65 42 16 40 188 141 137 136 139 144 143 144 142 142 142 142 142 142 140 135 133 134 125 144 160 150 125 124 111 90 50 71 117 103 115 122 126 117 123 125 128 122 120 119 117 112 96 70 9 72 124 113 114 113 141 146 135 117 95 117 103 95 116 115 106 109 108 98 94 112 129 132 138 145 149 159 165 151 166 181 175 182 169 149 146 135 115 137 154 155 139 97 92 92 87 76 82 79 77 103 170 140 140 140 140 139 140 140 143 143 143 143 143 143 141 136 132 133 127 130 148 147 124 115 113 84 33 84 102 93 111 102 99 96 86 80 83 78 89 95 101 99 94 48 5 55 115 101 105 103 139 142 141 144 153 148 152 153 151 155 156 164 170 170 175 183 186 193 196 192 194 195 180 162 157 169 170 163 164 170 171 166 160 156 156 161 151 152 170 168 155 155 164 165 178 188 174 139 143 143 143 143 144 143 142 143 143 143 142 143 141 137 132 132 128 125 148 147 129 119 110 104 86 111 114 102 108 100 95 85 66 70 77 68 61 49 28 26 31 27 20 36 106 90 101 101 140 143 142 140 146 150 150 149 152 156 154 157 161 161 163 165 170 173 174 175 179 183 175 159 152 158 166 166 165 172 175 171 167 165 166 161 157 167 175 181 161 152 158 153 157 164 172 139 143 143 143 143 142 143 143 143 143 143 143 143 141 137 131 131 128 126 148 150 124 109 132 143 133 123 128 118 127 115 108 117 113 116 113 103 102 85 56 44 44 91 76 34 119 115 103 100 146 148 147 147 153 156 156 156 158 161 161 162 163 163 161 162 168 168 166 166 167 169 179 108 94 104 97 98 98 100 105 111 110 106 105 105 97 124 181 171 156 130 119 110 106 110 115 138 142 142 142 143 142 142 143 143 143 143 143 142 138 137 134 134 126 121 147 141 89 63 82 105 132 129 117 113 115 100 95 97 89 102 94 103 105 103 107 83 67 108 74 20 97 105 104 101 151 158 154 154 156 158 157 157 158 159 159 159 160 160 159 159 160 162 162 162 164 164 169 112 109 112 109 110 110 108 108 112 110 109 112 116 122 123 139 135 135 134 129 116 100 104 104 136 142 142 142 143 143 142 143 143 143 143 143 141 137 138 136 136 126 118 146 143 95 81 71 74 91 112 118 105 87 88 87 84 84 100 87 90 83 87 96 88 80 97 76 26 97 101 100 97 153 162 155 158 157 157 158 158 158 158 158 158 158 158 158 158 158 161 163 163 164 162 161 114 110 110 111 111 111 112 113 112 110 113 119 124 130 136 134 137 141 138 137 129 114 98 93 137 143 142 143 142 143 143 143 143 143 143 143 141 137 138 136 135 126 118 147 162 117 87 84 95 89 84 111 110 102 104 97 87 82 82 83 85 89 83 77 85 93 89 77 27 91 97 94 95 153 162 156 157 158 158 158 158 158 158 158 158 158 158 158 158 158 161 163 163 161 157 156 113 115 115 115 116 116 113 111 113 116 118 122 129 134 138 146 146 140 134 138 137 129 116 103 138 143 142 143 143 143 143 143 143 143 143 143 141 137 138 136 133 131 112 150 169 113 88 101 110 105 95 83 115 111 97 98 87 83 87 90 83 82 86 84 87 87 79 72 33 57 88 86 85 154 163 155 158 158 158 158 158 158 158 158 158 158 158 158 158 158 159 160 160 159 156 149 119 121 121 119 118 118 116 118 104 102 119 129 129 136 143 141 139 136 135 139 138 134 136 122 139 143 143 143 143 143 143 143 143 143 143 143 141 137 137 136 131 134 109 140 155 116 105 103 109 107 106 100 88 122 122 111 101 90 100 103 91 96 95 94 92 97 101 100 102 88 88 86 86 159 163 156 158 157 157 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 156 147 122 122 122 119 117 117 116 116 68 81 118 128 132 142 144 138 138 138 138 137 137 138 140 130 139 143 143 143 143 142 142 143 143 143 143 143 141 137 137 136 131 131 116 131 117 94 87 88 101 109 113 121 97 107 143 119 97 99 144 150 110 86 90 90 78 76 93 93 94 90 93 79 88 168 160 159 158 157 157 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 156 147 121 122 122 119 117 115 126 101 34 96 132 143 146 153 145 137 138 137 137 137 138 134 132 123 140 137 140 143 143 143 143 143 143 143 143 143 141 137 138 136 132 131 120 119 73 71 75 82 92 97 99 104 118 112 126 138 117 127 157 154 138 103 108 109 102 97 104 107 111 107 106 101 121 164 158 159 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 159 155 146 122 122 123 122 121 120 132 76 84 153 138 141 145 140 141 139 137 137 136 136 137 125 106 101 132 133 139 143 143 143 143 143 143 143 143 143 141 137 138 137 132 133 113 134 135 135 131 136 138 136 136 138 142 141 143 150 144 132 136 148 152 152 152 152 153 154 152 150 149 150 155 162 163 157 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 158 160 154 145 123 123 123 122 123 123 119 108 123 79 91 137 146 146 143 129 137 146 144 148 124 118 87 80 135 134 139 143 143 143 143 143 143 143 143 143 141 137 137 136 131 132 114 142 141 137 143 148 152 151 151 149 147 147 146 150 152 142 140 148 150 148 148 148 147 147 149 150 150 149 151 156 157 156 155 155 156 156 156 156 156 156 156 155 157 158 158 158 158 158 158 158 158 160 154 145 123 123 123 122 123 128 126 70 15 18 49 70 84 110 130 140 144 138 118 119 162 130 82 77 140 134 139 143 143 143 143 143 143 143 143 143 142 140 136 134 132 132 116 145 143 147 147 148 148 146 146 146 146 146 149 150 155 155 161 158 159 160 160 161 163 163 162 161 162 161 149 160 164 170 166 170 164 164 168 172 171 169 171 172 165 160 161 162 158 158 158 158 158 159 156 143 126 119 132 122 125 92 48 13 25 93 121 101 71 48 43 71 101 124 137 80 73 148 118 55 139 134 139 143 143 143 143 143 143 143 143 143 143 141 136 133 133 133 116 145 145 147 148 148 148 148 148 148 148 147 151 155 150 155 145 147 142 135 136 135 133 135 129 122 124 119 151 168 168 122 119 119 124 122 131 133 133 132 138 136 140 171 157 157 158 158 158 158 158 158 157 142 123 120 127 109 43 10 17 21 52 121 113 120 119 104 76 45 31 43 89 143 85 44 136 107 139 133 139 143 143 143 143 143 143 143 143 143 143 141 136 133 133 133 116 145 145 147 148 148 148 147 147 147 147 146 153 138 41 29 19 22 19 14 15 14 12 13 11 10 10 10 91 153 93 11 8 8 27 28 13 21 19 16 20 12 56 175 164 159 157 158 158 158 158 158 157 142 122 129 85 19 17 22 21 17 46 80 73 80 105 131 138 100 56 41 39 51 105 81 54 131 144 139 142 143 143 142 143 143 143 143 143 143 142 138 137 135 132 132 115 144 144 146 147 147 147 148 148 148 148 146 154 138 20 12 12 12 12 12 12 13 16 15 17 15 17 23 18 25 15 24 23 19 18 18 18 25 39 31 28 26 41 172 166 158 157 158 158 158 158 158 157 141 133 78 6 16 23 24 21 19 41 64 67 65 85 112 103 103 78 32 41 43 32 74 57 47 142 143 143 143 143 143 143 143 143 143 143 143 141 137 137 136 131 132 115 144 144 146 147 147 147 148 148 148 148 147 152 136 22 15 13 13 13 13 13 13 13 12 14 14 19 24 21 19 24 23 17 14 12 12 14 22 37 33 25 23 37 169 162 157 158 158 158 158 158 158 157 145 112 17 22 21 24 22 25 25 30 46 55 65 55 45 27 38 45 43 30 33 29 33 81 64 141 142 142 142 142 142 142 143 143 143 143 143 141 137 137 136 131 132 115 144 144 146 147 147 147 147 147 147 147 146 151 130 15 11 13 13 13 13 13 13 13 13 10 17 25 17 15 15 17 18 11 10 14 12 16 23 25 33 31 28 39 157 162 159 158 158 158 158 158 158 156 150 79 6 27 19 23 23 18 26 28 23 44 96 45 22 47 32 31 23 43 54 38 34 56 121 140 143 143 143 143 143 143 143 143 143 143 143 141 137 138 135 131 130 116 143 147 147 147 147 147 147 147 148 149 145 146 126 16 9 10 10 10 10 10 10 11 15 14 14 17 13 13 13 14 14 12 12 13 13 15 21 24 31 27 26 32 150 160 157 158 158 158 158 158 159 159 148 79 11 20 21 23 23 20 26 33 38 29 111 125 54 86 105 87 100 115 90 59 54 39 64 143 143 143 143 143 142 143 143 143 143 143 143 141 137 138 134 131 129 117 142 148 147 147 147 147 148 148 150 151 145 145 126 18 8 9 8 8 8 8 8 10 16 16 13 12 13 13 13 13 13 13 13 13 14 9 11 28 25 13 24 30 148 159 156 158 158 158 158 158 159 162 144 94 17 18 21 22 21 29 22 28 40 38 118 156 132 101 105 138 144 137 101 61 59 55 58 142 143 143 143 143 143 143 143 143 143 143 143 141 137 138 134 131 129 117 142 148 147 147 147 147 148 148 150 151 145 145 126 18 8 9 8 8 8 8 8 10 16 16 13 13 13 13 13 14 14 13 13 13 13 13 15 13 28 34 26 29 149 159 156 158 158 158 158 158 159 162 142 113 32 18 25 24 23 22 31 37 44 87 140 131 138 147 146 138 134 140 122 72 68 59 52 140 143 143 143 143 143 143 143 143 143 143 143 141 137 137 135 128 130 116 142 148 147 147 147 147 147 147 148 148 146 146 129 19 8 8 8 8 8 8 9 12 14 14 12 13 8 10 16 9 16 17 11 13 12 14 15 8 17 32 27 25 144 162 156 158 158 158 158 158 159 163 140 97 61 19 16 16 22 27 15 60 121 116 139 140 137 122 136 141 139 129 128 85 56 36 28 138 142 142 142 142 142 142 143 143 143 143 143 141 137 137 136 127 131 116 143 149 148 148 148 148 147 147 147 147 147 147 132 19 7 8 8 8 8 8 10 13 13 13 12 13 9 9 19 11 11 20 16 12 13 13 12 14 12 21 27 24 142 164 156 158 157 157 158 158 159 163 139 85 84 40 9 11 12 20 14 41 110 107 140 129 117 115 103 130 130 130 120 88 61 35 29 138 142 142 142 142 142 142 143 143 143 143 143 141 136 137 136 127 131 116 143 149 148 148 148 148 147 147 147 147 147 147 130 18 8 7 8 7 8 8 10 13 13 13 12 12 16 14 16 29 15 5 11 13 13 12 13 13 13 23 27 24 142 164 156 157 158 158 158 158 159 163 139 88 75 78 38 20 18 15 19 18 55 65 53 41 73 144 142 121 123 129 126 100 65 36 29 138 142 142 142 142 142 142 143 143 143 143 143 143 143 140 135 128 134 112 136 146 147 147 147 147 147 147 147 147 147 147 135 21 8 9 13 12 7 8 10 13 13 13 13 14 7 29 97 120 108 51 17 8 13 14 14 12 15 24 24 20 141 161 155 158 158 158 158 158 159 160 137 78 71 77 63 42 26 16 18 22 24 27 10 44 119 126 118 114 127 132 127 108 55 29 31 138 142 142 142 142 142 142 143 143 143 143 143 144 146 141 135 128 135 110 133 145 147 147 147 147 147 147 147 147 147 147 137 22 8 10 16 14 7 8 10 13 13 13 13 14 8 33 108 117 122 129 115 80 19 10 5 20 23 19 19 16 141 159 155 158 158 158 158 158 159 159 136 72 72 66 60 49 33 16 10 15 20 23 22 38 70 83 101 112 127 127 129 121 52 29 25 138 142 142 142 142 142 142 143 143 143 143 143 144 146 141 135 128 135 111 134 146 148 148 148 148 147 147 147 147 147 147 136 22 8 10 16 14 7 8 10 13 13 12 13 12 13 30 90 115 115 125 126 140 85 10 13 14 14 15 14 14 142 160 155 158 157 157 158 158 159 159 136 71 66 62 56 50 43 28 18 13 18 25 21 30 78 122 144 133 126 119 134 108 38 23 43 141 142 142 142 142 142 142 143 143 143 143 143 143 144 142 139 130 135 106 137 146 147 148 148 147 147 147 147 147 147 147 137 22 8 9 10 10 8 8 10 13 13 13 12 13 11 15 82 119 115 119 126 124 124 33 2 16 10 13 12 12 138 162 154 158 157 157 157 158 154 155 135 65 63 62 54 44 36 33 30 25 19 17 24 55 99 111 128 131 124 130 121 51 43 44 36 142 142 142 142 142 142 142 143 143 143 143 143 142 142 143 140 131 135 104 138 147 147 148 148 148 147 147 147 147 147 147 137 22 8 8 8 8 8 8 10 13 13 13 12 13 13 21 76 119 124 120 119 121 126 52 0 19 14 12 13 12 137 164 154 158 157 157 158 158 154 156 132 59 58 59 58 51 41 34 29 25 20 17 18 90 137 129 138 133 131 116 51 57 97 99 90 142 143 143 142 142 142 142 143 143 143 142 143 143 143 143 140 131 135 104 138 147 147 148 148 148 147 147 147 147 147 147 137 22 8 8 8 8 8 7 10 13 13 12 13 14 11 16 28 79 122 127 121 127 133 89 9 7 11 13 12 12 138 163 154 158 158 158 158 157 159 161 128 54 54 54 53 47 38 33 30 24 25 21 19 34 63 47 107 91 59 24 53 114 115 131 134 138 139 139 140 142 142 142 143 143 143 143 143 143 142 138 135 132 135 104 138 147 147 148 148 148 147 147 147 147 147 148 141 29 9 8 8 8 8 8 9 9 9 12 10 11 18 26 8 18 77 108 93 96 106 92 37 8 7 15 12 14 139 168 154 158 158 157 157 157 156 155 127 45 50 56 55 46 34 28 28 26 25 22 22 17 20 17 14 15 13 40 104 128 132 134 128 136 135 135 137 142 142 142 142 142 142 142 142 142 141 136 133 132 135 104 138 147 147 148 148 148 147 147 147 147 147 148 142 32 9 8 8 8 8 8 8 7 7 11 4 0 43 94 7 4 87 88 10 34 90 93 41 16 14 14 16 27 140 165 154 158 158 158 157 157 155 153 127 43 48 51 51 46 39 30 26 28 24 21 23 18 20 19 20 26 42 74 115 127 134 132 133 138 139 139 140 142 142 142 143 143 143 143 143 143 141 136 133 132 135 104 138 147 147 148 148 148 147 147 147 147 147 148 142 31 9 8 8 8 7 7 7 7 7 8 39 38 48 84 26 10 134 75 12 56 79 115 61 1 17 11 19 46 139 156 154 159 157 158 158 158 156 154 128 45 47 40 40 43 43 34 26 28 24 22 23 17 19 19 17 17 49 98 132 130 132 133 133 138 139 139 140 142 142 142 142 142 142 142 142 141 138 137 135 131 135 102 136 151 148 148 148 148 147 147 147 147 147 147 142 32 7 9 8 8 7 7 7 7 7 6 49 117 128 91 20 36 130 118 78 111 121 122 68 3 19 34 47 52 136 159 155 158 157 157 158 158 152 157 131 40 49 46 39 38 40 33 27 28 29 21 19 26 30 19 38 32 80 126 130 129 126 125 125
## 6406                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  151 145 148 150 61 10 17 21 16 10 4 81 139 122 50 7 11 8 4 15 114 179 173 123 147 159 128 132 132 142 154 137 119 121 112 112 111 93 95 91 86 72 64 67 56 53 69 70 50 37 14 19 25 25 25 25 25 26 28 30 29 28 28 30 26 29 28 66 37 4 8 16 26 27 26 34 124 47 12 20 15 17 22 9 11 30 67 113 103 106 112 81 111 166 164 164 151 150 153 151 69 7 19 18 17 9 6 65 137 123 56 7 14 5 4 12 25 46 45 40 36 35 31 27 27 23 19 17 17 13 13 10 9 7 7 6 6 5 6 5 7 7 8 12 7 4 9 28 23 24 24 25 25 26 28 30 30 31 30 32 28 30 33 66 31 6 9 20 28 28 25 32 129 42 17 21 17 18 20 11 11 31 68 109 109 107 115 79 116 164 164 157 153 151 152 153 72 8 20 20 18 9 3 60 126 113 55 12 10 6 8 9 7 0 3 0 2 3 1 4 4 5 6 8 9 9 9 9 10 10 11 10 11 9 11 11 12 12 10 13 10 5 8 26 24 25 25 24 25 26 28 29 30 31 30 33 29 32 32 64 38 4 8 20 29 30 26 33 129 43 17 20 17 19 21 11 11 32 70 109 109 108 114 80 119 166 164 158 151 150 151 157 77 9 21 21 18 10 1 51 115 106 49 8 11 9 10 9 9 9 9 8 8 9 8 10 10 10 10 10 10 11 10 10 10 9 9 12 11 10 9 9 9 10 8 10 12 9 7 24 24 25 25 24 25 26 29 30 31 32 31 33 33 33 30 62 45 0 8 20 30 31 27 34 129 43 18 21 18 20 22 10 11 34 73 110 110 108 115 82 121 167 164 158 153 152 151 163 85 9 20 22 19 11 1 46 107 104 49 6 16 11 7 7 9 5 5 6 7 6 8 7 7 7 8 8 7 7 6 7 8 9 10 9 9 8 8 9 9 13 11 15 19 15 3 21 24 25 25 24 25 26 29 30 31 32 32 32 31 33 32 69 69 9 7 21 31 34 29 35 129 43 17 21 20 21 23 11 11 36 76 109 111 109 114 84 124 170 165 159 155 155 153 169 91 10 19 20 18 13 2 44 106 107 54 6 13 10 9 8 8 6 6 12 13 13 17 18 19 20 22 23 25 28 29 31 33 36 36 38 38 39 40 42 44 49 46 51 54 42 17 25 24 25 25 24 25 26 30 31 32 33 32 31 29 32 34 75 90 18 5 22 33 34 31 37 129 43 17 22 20 22 23 11 12 39 78 109 112 111 115 85 127 172 165 159 154 154 155 171 94 10 18 20 18 13 3 44 107 109 75 45 45 48 56 54 52 51 48 55 56 55 59 60 62 61 61 61 63 64 64 64 64 64 64 65 64 63 63 64 64 65 65 69 70 53 17 24 25 25 25 24 25 26 30 31 32 33 33 31 31 32 32 73 91 14 5 22 34 35 32 38 129 42 18 22 21 23 25 12 12 39 79 109 112 111 114 85 130 172 167 158 158 155 158 166 96 7 22 23 21 12 3 42 107 111 82 70 70 68 66 68 67 64 66 64 65 64 62 62 67 65 64 65 61 62 61 63 63 65 66 63 63 64 65 66 68 64 64 68 67 65 9 29 25 26 24 25 26 28 29 29 30 32 32 32 32 32 33 82 88 13 7 23 38 40 33 42 133 37 21 20 22 22 29 8 14 37 85 113 111 113 115 82 135 174 178 152 157 154 157 168 98 10 21 22 23 15 4 41 104 110 82 69 70 69 69 71 72 71 73 74 76 78 78 75 82 83 85 90 86 87 87 88 89 90 90 95 97 98 100 100 100 106 107 109 109 95 21 29 26 26 25 25 27 28 30 30 31 32 32 32 33 31 34 83 88 13 5 25 40 41 33 44 135 38 20 20 23 22 27 7 14 38 85 112 111 112 114 82 135 173 175 150 160 161 158 169 104 12 21 23 24 14 2 42 104 111 100 106 105 106 108 109 110 114 115 117 119 122 122 121 126 124 126 127 120 119 119 121 122 122 122 128 129 130 131 131 131 135 133 134 135 115 23 26 26 26 26 25 27 29 30 30 31 32 33 33 34 34 35 83 90 14 5 26 42 44 37 47 135 38 22 21 22 23 26 6 15 40 86 111 111 112 114 84 137 173 173 148 159 162 158 171 110 14 20 23 24 11 0 42 105 116 113 123 123 125 127 128 127 126 127 128 129 130 130 131 132 130 130 129 120 128 129 130 131 131 132 133 133 134 133 133 133 136 132 132 134 117 22 25 28 27 26 26 28 29 30 31 31 33 33 33 35 35 35 84 90 15 4 28 45 44 39 50 135 38 22 21 23 23 25 6 17 43 88 111 111 112 114 86 140 173 172 148 155 158 155 172 114 17 20 23 25 10 0 41 104 120 117 122 122 123 125 125 125 127 127 128 128 128 128 129 130 130 134 131 120 130 132 133 133 134 135 136 137 136 136 136 136 136 131 132 136 118 23 27 27 28 27 26 28 30 31 31 32 33 33 34 35 35 36 86 91 15 3 29 49 47 41 53 136 39 22 22 23 24 25 7 20 46 91 113 114 113 115 89 143 175 173 151 157 160 156 175 117 19 20 22 27 13 1 42 102 118 119 124 126 126 126 125 125 127 128 129 130 132 132 134 135 135 137 134 121 131 133 134 136 137 137 139 139 140 138 138 139 138 137 141 142 121 25 27 28 27 26 28 28 29 31 32 33 34 34 33 36 36 37 86 93 16 3 30 51 48 43 56 136 39 23 22 25 24 24 8 23 49 93 115 115 115 117 91 145 175 174 155 159 159 159 170 121 18 18 27 25 13 7 34 104 123 120 124 125 126 127 127 128 130 131 131 132 133 134 135 135 135 134 139 120 134 136 137 137 139 140 142 142 143 143 143 144 143 143 139 144 124 22 29 28 28 27 28 29 31 32 33 32 31 33 34 37 35 36 94 94 16 6 31 54 49 48 60 138 32 24 22 28 27 26 10 19 51 95 115 117 120 110 91 147 176 178 159 159 158 159 170 125 21 20 29 27 13 7 32 105 125 121 126 127 128 130 130 130 131 132 133 134 135 136 137 138 138 138 142 123 136 139 140 141 141 143 144 144 143 146 145 147 147 147 141 147 128 24 30 29 27 27 28 30 31 32 34 34 32 33 33 37 35 37 95 94 16 7 33 57 52 50 63 135 31 25 23 29 26 25 10 19 53 95 115 117 122 109 91 148 177 181 161 163 161 161 172 129 27 22 29 30 15 7 32 104 125 122 127 128 130 131 131 131 133 134 135 137 137 138 139 141 143 142 145 126 141 144 145 146 146 146 147 147 148 148 148 149 150 151 146 152 130 26 31 29 28 28 29 31 32 33 35 34 33 33 34 37 37 38 96 93 14 5 36 62 54 52 66 136 33 29 28 32 27 27 12 20 54 96 116 119 121 110 92 149 179 182 162 163 163 161 173 133 32 24 29 34 17 7 31 103 126 123 127 129 131 133 133 133 135 136 137 139 141 141 141 143 145 143 146 127 142 145 145 146 147 148 151 152 152 152 153 152 152 155 149 153 132 26 30 29 28 28 29 31 32 33 35 35 33 34 34 39 38 40 98 93 13 4 39 67 58 54 68 135 32 28 28 32 28 28 12 21 55 97 117 118 122 111 93 151 180 183 161 161 161 161 173 136 36 23 29 37 19 7 30 102 126 124 128 130 132 134 135 135 136 138 140 142 144 144 145 148 150 148 151 131 145 148 148 149 150 151 154 155 156 155 156 156 157 157 152 157 134 26 30 30 29 29 30 32 32 33 35 35 34 34 35 38 38 41 98 93 12 5 43 73 61 58 73 133 29 25 24 30 28 28 13 22 55 97 118 119 122 111 94 152 182 184 162 162 161 159 172 137 37 23 27 39 21 7 29 102 126 125 129 131 134 136 137 137 137 139 141 143 146 147 150 153 156 155 156 136 148 151 152 153 153 154 156 156 157 156 157 157 159 162 156 161 136 28 32 30 29 29 30 32 33 34 36 35 34 35 35 38 40 42 100 94 12 9 48 78 67 64 78 139 33 27 25 34 33 30 14 22 56 98 119 120 122 112 94 153 183 186 164 160 160 158 167 147 40 20 29 37 20 6 23 104 124 126 130 131 133 135 137 137 141 139 142 152 149 148 152 158 159 158 159 134 151 154 155 157 158 160 162 162 162 162 163 162 167 161 165 162 143 27 31 30 29 29 30 32 33 33 34 34 34 34 35 39 38 41 99 91 15 6 53 83 73 65 83 137 31 28 31 32 35 32 11 26 57 105 119 118 127 109 98 155 185 188 166 164 165 162 168 152 44 21 32 36 20 6 22 106 129 130 129 132 133 135 137 139 145 144 143 144 144 156 156 154 153 160 167 140 154 159 159 160 162 162 166 165 164 165 165 165 167 160 165 160 143 33 35 29 29 29 30 32 33 33 34 34 35 34 36 40 38 42 99 91 16 7 58 88 78 68 86 136 29 28 32 32 35 32 11 28 58 107 119 115 123 104 97 155 185 188 166 166 167 167 171 154 47 23 33 37 22 9 20 105 131 132 132 133 136 138 138 142 143 139 152 155 153 155 156 160 163 166 167 134 156 161 161 162 164 164 164 166 166 168 169 171 172 170 175 173 145 24 28 32 29 29 31 33 33 33 34 34 35 34 36 40 39 42 100 93 18 8 64 94 83 74 91 135 29 30 33 32 35 33 11 29 61 107 120 117 124 105 98 159 188 189 166 167 170 169 172 155 48 22 30 40 25 10 15 100 126 127 128 130 132 134 134 139 139 139 149 144 145 138 136 149 149 145 151 132 143 147 148 148 149 149 153 155 156 155 158 158 159 153 149 154 134 29 30 32 30 30 31 33 34 33 34 35 37 36 38 39 40 43 101 94 18 7 69 99 86 79 94 133 30 30 34 34 36 35 11 32 65 108 121 122 128 106 100 160 188 188 166 174 178 174 173 157 50 22 29 41 29 12 12 97 125 125 125 127 129 131 132 131 127 141 117 76 90 91 99 135 155 151 150 129 149 151 152 152 153 153 156 155 154 153 151 151 148 144 140 145 122 29 31 32 31 31 32 34 34 33 34 35 37 37 38 40 40 43 103 96 18 7 75 106 93 85 98 132 29 31 36 35 36 37 11 34 68 109 122 123 127 104 99 158 183 187 169 176 180 179 178 159 54 24 30 43 29 13 11 98 130 132 134 136 138 140 141 142 146 157 122 76 79 44 46 72 81 78 82 60 61 63 63 63 61 62 53 51 50 51 52 52 46 50 39 38 38 11 35 31 31 31 32 34 34 32 34 36 38 38 39 41 41 44 102 97 21 8 80 111 98 90 102 134 30 31 37 36 36 38 11 36 70 108 122 123 125 103 99 162 188 191 171 175 178 176 171 166 58 23 33 39 31 10 9 99 129 132 137 137 139 140 141 145 147 148 123 87 75 18 5 4 7 8 17 11 20 22 22 25 30 34 46 53 57 62 67 71 77 94 99 99 91 15 34 32 32 32 33 33 33 33 35 36 37 39 40 42 42 42 108 99 16 10 88 119 106 95 106 130 29 33 37 36 41 39 10 41 72 114 121 119 130 103 105 164 186 191 177 179 180 177 167 169 62 21 35 38 34 12 9 99 129 132 135 135 138 139 140 139 139 157 130 74 73 57 72 110 147 151 148 133 148 153 158 162 165 168 168 170 173 174 176 181 177 178 179 178 160 34 35 33 33 32 33 33 33 34 36 36 36 39 40 42 43 43 112 100 14 14 97 125 113 99 110 126 27 32 34 34 40 40 10 44 73 117 120 117 132 103 106 165 184 189 182 178 179 180 171 172 64 21 37 42 37 13 9 101 131 133 136 136 139 141 142 142 143 154 133 87 93 99 103 153 181 179 178 154 169 175 177 178 180 186 188 191 192 189 188 189 195 193 187 191 170 29 29 35 33 33 33 34 33 34 36 36 36 39 40 42 43 45 112 100 15 17 108 133 117 106 115 128 29 35 37 35 40 41 10 46 75 116 120 118 131 103 109 166 185 189 184 178 177 180 173 177 69 20 38 47 39 15 11 101 133 135 138 138 141 143 142 150 158 147 153 154 159 160 166 173 168 169 178 149 170 177 178 176 179 187 187 188 191 191 192 194 194 191 189 194 173 33 32 35 34 33 34 35 34 34 36 36 36 39 41 42 42 47 114 98 14 19 120 141 122 114 118 131 33 39 41 38 43 43 11 48 78 115 119 119 131 102 111 170 187 188 186 177 178 180 172 180 74 20 36 47 40 15 10 101 132 135 139 139 142 144 146 148 146 146 155 162 167 172 175 171 172 181 183 147 160 176 187 186 185 188 195 197 199 199 199 202 201 198 197 197 174 38 34 35 34 34 34 35 34 34 36 36 36 39 41 41 41 50 116 97 15 21 131 150 127 121 122 127 31 39 42 39 43 44 11 50 82 115 120 120 131 102 113 173 188 186 186 173 172 177 171 183 77 20 33 47 40 15 9 100 131 135 139 140 143 145 147 147 154 161 156 161 160 166 174 176 172 150 133 109 106 124 142 156 172 187 195 197 199 198 200 203 205 203 201 205 182 42 35 35 35 34 35 36 35 34 36 36 36 39 41 42 41 52 115 96 15 26 141 158 132 127 125 126 30 41 44 42 46 46 11 51 83 114 119 121 130 103 114 175 189 185 185 168 166 172 170 183 81 19 31 39 44 13 9 98 135 136 141 143 144 147 150 150 151 159 158 163 170 171 152 127 108 76 51 46 44 58 72 85 116 151 180 199 203 207 205 204 212 208 205 211 188 38 31 37 34 33 34 36 36 35 37 37 37 40 41 41 42 49 123 102 14 30 150 165 142 133 131 128 33 42 46 46 49 51 10 54 85 115 120 121 131 102 115 178 187 181 187 169 168 171 170 181 83 19 31 31 48 13 10 97 139 139 143 144 145 148 152 155 157 161 165 167 169 144 104 59 41 32 18 23 20 29 34 37 54 75 123 169 192 212 209 209 216 212 211 216 193 42 33 37 34 33 34 36 36 35 37 37 37 40 41 41 44 47 132 108 14 34 159 171 151 138 136 127 34 43 45 47 48 55 9 56 85 118 123 123 133 101 119 180 187 181 193 170 171 172 170 183 86 19 31 34 50 13 10 96 138 139 143 145 146 148 151 155 160 161 165 167 143 88 60 28 17 18 17 23 23 25 25 22 21 21 53 99 151 201 214 216 218 216 214 219 196 44 33 37 34 33 34 36 36 36 37 37 37 39 41 43 44 51 132 106 15 37 165 177 157 144 139 126 34 44 46 47 49 55 9 56 86 119 125 127 134 101 121 182 191 185 195 173 173 175 170 186 91 20 31 37 53 15 8 94 139 142 144 145 148 149 151 156 159 164 169 136 89 51 29 24 30 40 38 34 36 41 40 29 22 21 23 38 92 171 217 218 219 220 218 222 199 46 33 37 34 33 34 36 36 36 38 37 36 38 41 42 42 52 131 103 14 40 172 184 163 151 140 126 35 46 48 47 51 57 10 58 88 121 126 128 132 99 123 183 192 188 198 173 174 177 170 188 96 23 30 39 57 17 8 92 140 144 146 147 148 151 154 157 162 168 154 87 44 38 18 39 57 66 61 51 57 66 62 45 33 35 26 12 37 117 206 226 220 223 222 224 202 48 34 37 34 33 34 36 37 38 38 37 36 37 40 41 40 52 132 99 12 44 180 191 170 156 142 124 37 49 48 48 53 58 12 63 92 123 124 127 128 98 127 185 193 188 198 173 174 177 169 190 99 24 30 40 58 18 7 91 140 145 147 146 148 151 155 157 164 165 117 43 28 28 33 61 72 76 83 85 91 96 91 76 58 50 40 24 18 59 168 228 223 223 222 227 202 48 35 37 34 33 34 36 37 38 38 37 36 37 41 41 41 56 133 99 14 49 187 199 176 162 145 123 37 51 49 49 54 57 14 65 96 124 121 126 128 101 133 189 194 189 197 173 173 178 174 189 104 24 32 42 66 19 9 89 141 146 146 147 149 153 156 157 165 157 68 21 20 20 47 72 80 84 98 115 115 114 110 102 83 61 55 44 16 34 155 225 223 224 223 227 203 48 34 37 33 33 34 36 37 38 39 37 36 35 40 44 41 55 136 98 13 52 193 199 183 167 149 122 41 50 51 50 56 61 13 76 100 124 121 125 131 102 138 193 195 187 199 173 174 179 182 187 114 27 35 43 78 21 10 83 142 144 147 147 149 153 158 158 167 147 33 20 14 20 41 75 90 103 113 132 131 134 128 119 105 84 66 53 17 32 171 228 225 228 226 231 209 50 34 35 31 32 35 36 36 37 39 37 34 34 37 43 40 50 142 97 11 57 201 200 193 175 153 119 47 49 52 53 58 67 9 91 105 123 125 124 133 99 137 193 196 182 204 172 173 177 178 187 116 29 36 44 78 24 8 79 140 146 149 149 151 155 158 160 166 139 31 17 14 21 44 82 89 102 119 133 138 147 143 128 114 104 78 60 30 17 153 230 227 228 225 233 213 52 34 35 31 32 35 36 36 37 39 37 35 35 37 42 40 53 145 101 15 62 210 213 201 183 157 118 47 51 53 54 57 67 10 91 107 123 124 124 130 98 140 193 195 184 204 172 173 177 174 187 121 31 36 42 78 28 9 76 140 146 148 148 150 153 156 159 167 137 32 19 13 14 31 80 95 107 112 113 129 134 133 122 107 100 86 67 42 15 153 235 229 228 225 230 211 49 32 36 31 32 35 36 36 37 39 38 35 36 38 42 41 53 145 99 12 66 219 223 208 189 157 118 47 52 54 54 57 69 11 96 111 122 125 127 132 102 146 197 200 187 205 172 175 176 172 187 126 34 35 38 75 34 12 73 140 148 150 150 152 154 157 160 172 142 39 21 6 11 43 85 102 113 123 133 126 123 126 122 108 98 88 66 40 26 166 231 231 235 228 231 211 57 37 35 31 32 35 36 36 37 40 38 37 38 39 43 42 53 143 96 7 69 226 229 209 193 158 120 48 51 53 55 58 71 14 100 114 122 127 129 130 102 147 198 199 189 208 174 176 178 172 188 129 34 33 37 77 39 15 69 134 143 147 145 147 150 152 154 160 137 36 17 12 29 68 81 83 85 89 117 147 148 150 136 110 98 79 55 25 37 177 213 199 201 198 198 181 37 31 36 30 32 35 36 35 36 40 38 37 39 39 42 44 56 146 99 9 72 232 234 209 196 158 123 48 49 51 55 62 73 15 104 117 121 128 129 129 102 150 197 199 190 209 175 177 179 173 187 134 34 32 37 80 49 16 68 132 137 135 134 136 139 140 141 148 128 40 35 22 52 92 106 110 102 89 106 128 134 132 108 84 84 86 56 8 52 190 204 200 205 203 203 188 47 35 35 32 32 35 36 35 36 39 40 37 39 40 42 44 58 148 102 13 76 236 235 213 196 161 125 49 47 50 56 62 74 16 105 119 122 129 131 130 104 155 200 200 192 210 175 178 181 180 183 142 33 35 37 75 64 13 74 138 150 148 149 151 153 155 157 167 140 95 104 36 73 115 127 105 95 101 111 119 123 115 108 105 121 131 85 1 107 223 224 226 223 220 225 207 51 32 35 34 34 35 36 37 37 38 39 38 39 39 45 42 56 143 98 14 81 243 230 223 194 166 124 52 44 50 58 58 71 19 107 119 127 129 131 128 103 158 200 196 191 211 175 177 181 178 184 145 36 34 37 74 66 14 70 139 148 147 148 150 152 153 160 147 104 118 113 66 84 114 142 151 152 133 115 106 103 98 105 91 66 109 88 18 158 231 221 222 222 218 223 206 50 33 36 34 34 36 37 37 38 39 39 39 39 39 45 43 55 141 98 16 83 244 230 222 194 166 123 53 47 52 58 59 71 17 105 118 126 129 132 130 104 158 202 201 193 210 174 177 181 177 185 150 38 33 36 71 70 15 67 141 147 148 148 149 152 153 161 95 28 111 121 84 79 131 127 116 122 117 110 93 91 94 129 140 121 119 78 48 165 225 222 223 225 219 224 207 51 34 36 35 35 36 37 38 38 39 40 39 40 40 47 43 54 140 98 15 86 244 230 222 193 166 121 54 49 53 58 59 71 18 106 119 127 130 132 130 107 163 206 202 194 212 174 176 182 176 186 155 40 32 36 68 77 15 62 143 146 148 148 149 152 153 161 70 14 111 122 91 89 116 126 129 130 107 87 93 89 78 109 137 133 124 68 82 122 208 225 225 225 220 225 208 52 34 37 35 35 36 37 38 38 40 40 39 40 40 46 44 56 141 97 14 88 244 230 222 195 168 117 53 50 54 57 59 71 18 106 120 129 132 131 129 109 166 206 198 194 216 173 175 183 175 187 160 42 30 36 66 80 17 58 144 144 149 148 149 151 154 156 66 22 100 113 79 88 108 128 139 138 109 94 108 104 79 101 129 124 111 79 112 110 207 228 226 226 221 226 209 54 36 37 36 36 37 38 39 39 40 41 40 41 41 45 44 58 143 95 11 88 245 230 222 196 171 114 51 51 53 55 56 69 16 104 118 127 131 132 130 106 161 201 195 192 214 174 175 182 175 187 163 43 30 36 65 83 17 55 146 144 148 147 149 152 154 155 77 13 78 118 82 83 112 125 127 121 106 108 116 114 95 116 139 123 94 101 126 131 215 227 226 227 222 226 209 55 36 37 36 36 37 38 39 39 40 41 40 41 40 45 44 60 145 95 9 86 243 230 223 197 171 111 50 51 52 54 56 66 13 101 115 124 127 135 132 102 154 197 196 191 210 181 179 180 183 187 174 40 30 38 58 82 14 50 147 143 147 147 147 150 155 156 69 10 32 44 65 93 110 128 129 108 98 104 113 95 96 119 134 124 83 104 122 172 228 221 223 224 222 226 207 59 33 38 37 36 37 37 38 39 40 41 40 42 40 43 44 60 156 97 5 84 241 228 220 185 164 109 47 48 49 52 57 61 16 93 114 122 129 130 132 103 163 194 200 192 211 182 180 179 180 184 174 39 25 35 57 78 12 49 146 145 149 148 150 153 157 158 100 58 39 22 59 97 102 126 122 96 92 113 127 100 84 107 126 110 84 83 124 211 233 216 222 224 220 224 205 58 32 39 37 36 37 37 38 38 40 41 40 42 40 46 45 61 155 94 3 84 238 225 217 183 163 107 45 48 49 52 54 59 18 91 114 125 132 132 131 105 165 194 200 191 210 181 181 179 178 183 176 43 24 34 54 72 11 46 142 146 149 150 151 153 156 167 164 146 78 18 60 99 96 127 125 96 98 125 109 98 84 102 118 102 73 116 172 223 215 218 222 220 218 222 204 57 32 39 37 36 37 37 38 37 39 40 42 42 42 47 44 62 155 92 4 85 234 222 215 181 165 104 43 46 47 50 53 54 19 87 116 128 135 134 131 106 168 195 200 194 214 182 185 183 179 184 183 49 24 34 52 64 10 44 139 148 150 149 151 154 157 160 162 176 91 16 58 89 95 123 125 98 89 105 110 95 82 113 117 90 94 193 224 217 214 223 219 218 216 219 202 56 33 39 37 36 37 37 38 37 39 40 42 43 42 46 41 63 156 92 7 85 230 220 212 180 168 102 40 45 46 48 51 49 19 83 116 130 134 136 131 106 172 195 199 196 215 184 187 185 180 186 187 54 24 35 49 57 9 40 136 149 147 149 149 153 157 157 159 181 93 21 60 77 94 113 122 110 96 92 85 83 95 134 113 75 147 225 221 212 220 213 212 217 212 216 199 55 32 39 37 36 37 37 37 36 39 40 42 43 43 47 40 64 156 90 8 83 225 215 206 175 168 99 37 44 45 46 49 44 21 79 113 129 132 138 131 108 175 195 200 194 212 179 182 184 179 185 189 57 22 33 47 54 8 39 134 149 146 147 149 151 155 160 166 175 88 30 69 75 72 88 111 126 121 110 100 112 115 111 82 107 204 219 206 217 213 213 217 215 212 214 198 55 32 39 37 36 37 37 37 35 38 40 42 44 46 50 42 66 155 87 5 80 219 211 202 170 165 97 35 42 44 46 48 42 21 75 112 128 128 137 130 108 177 195 200 196 213 182 184 186 185 185 191 61 19 35 47 46 12 36 138 144 147 147 150 152 157 155 158 173 99 61 96 88 69 64 93 118 118 124 113 121 108 76 78 81 119 186 214 212 214 212 213 210 206 215 196 54 33 38 36 36 37 38 39 39 41 42 41 43 43 49 45 62 157 82 2 81 219 203 197 170 168 91 36 44 43 47 50 37 25 68 107 129 130 136 124 104 182 193 198 193 214 181 183 185 186 185 192 65 22 35 47 44 11 34 136 148 148 136 147 152 151 160 164 143 81 64 97 97 82 63 80 107 121 137 122 112 73 52 79 67 37 97 176 214 204 209 208 201 200 206 190 58 37 37 36 36 37 38 39 39 41 42 41 43 43 48 46 63 158 82 1 77 213 196 190 165 167 89 34 43 42 47 50 36 26 68 108 130 131 137 124 105 184 194 198 194 214 182 185 185 184 184 193 71 25 36 46 44 10 30 134 144 147 147 151 153 156 159 130 92 55 65 95 111 102 71 69 87 102 115 109 89 57 61 97 80 46 53 84 161 208 199 198 205 203 205 188 51 33 38 36 36 37 38 39 39 41 42 41 43 42 46 45 65 159 80 0 74 203 189 183 156 165 86 31 41 40 45 49 35 26 70 110 132 132 138 126 108 186 196 196 194 216 182 187 185 184 183 195 77 28 36 46 43 10 25 131 142 146 145 140 155 151 120 87 95 79 82 92 119 113 89 75 76 77 76 72 62 78 97 106 90 86 99 61 101 206 209 195 201 192 196 187 53 33 38 36 36 37 38 39 39 41 42 41 43 42 45 45 64 159 80 0 71 193 182 176 149 164 83 28 38 38 44 49 33 23 69 111 133 133 137 126 110 189 196 195 194 218 180 185 185 186 183 196 82 28 36 46 42 9 21 127 142 144 147 151 149 90 51 101 109 87 103 106 119 125 119 105 99 96 90 78 80 114 125 110 94 109 110 108 109 111 164 198 200 190 188 181 53 33 38 36 36 37 38 39 39 41 42 41 43 43 45 44 62 157 80 0 68 184 174 169 141 163 80 25 35 37 43 48 29 21 68 110 133 132 136 125 110 189 196 192 194 220 182 187 188 188 183 197 85 29 34 46 41 8 19 125 145 145 146 141 101 75 78 86 107 71 102 113 118 131 136 124 118 121 121 117 121 132 126 118 97 96 113 83 74 94 92 124 173 197 197 180 47 31 39 36 36 37 38 40 39 40 41 42 43 43 47 43 60 156 80 1 64 177 169 164 134 159 78 23 34 36 43 47 27 17 67 109 132 132 134 124 109 189 196 190 194 221 184 183 180 182 188 198 92 22 39 36 42 7 15 113 151 137 111 90 75 77 100 97 61 38 94 125 126 136 138 134 128 128 129 127 140 141 130 123 108 127 109 99 62 96 101 79 100 98 163 156 52 34 40 36 37 37 37 39 41 42 43 42 44 43 45 45 60 154 79 2 62 167 159 152 127 163 74 26 37 38 38 47 24 24 59 112 134 137 138 125 111 188 193 192 197 221 186 182 185 185 185 193 91 19 30 51 35 10 14 112 108 71 76 104 96 69 97 91 68 61 106 130 137 139 142 141 137 135 137 142 146 138 132 125 120 131 120 112 106 87 58 95 96 65 58 118 51 29 37 37 39 39 38 41 42 42 43 42 44 42 44 45 60 152 75 0 59 160 153 145 123 162 70 26 33 33 33 43 23 28 59 115 136 138 141 126 113 188 189 191 197 220 186 185 187 184 179 193 100 26 27 45 38 5 32 78 78 72 90 121 137 102 118 87 109 110 132 124 134 143 144 144 144 143 146 148 146 142 145 126 119 124 144 125 99 90 84 75 73 96 70 42 56 55 40 36 33 36 38 39 41 42 43 44 44 44 46 43 59 150 70 0 56 152 144 138 117 161 68 23 29 29 31 40 19 24 59 116 135 135 143 129 113 185 186 188 194 217 183 186 182 180 180 197 111 30 38 40 50 31 55 42 66 84 80 97 148 129 103 115 130 139 153 120 158 149 149 149 150 151 151 157 153 145 137 90 75 115 119 111 110 97 71 62 81 61 99 100 96 94 75 48 32 38 42 39 41 41 42 42 45 44 46 42 60 149 67 0 54 143 134 129 112 163 67 24 29 27 33 40 15 21 56 114 133 133 142 130 112 183 183 183 191 218 182 187 183 186 186 200 116 31 47 54 56 81 68 55 50 63 72 117 146 73 73 108 124 116 95 123 146 155 155 157 160 161 160 162 159 147 142 94 109 159 126 67 91 59 70 114 110 76 85 118 86 111 101 70 37 34 36 36 41 40 41 44 45 45 46 41 61 150 66 0 52 135 124 121 106 164 66 25 30 28 34 38 16 22 56 114 135 136 142 128 112 183 182 178 191 220 184 191 185 189 188 198 116 29 62 59 88 105 115 103 83 100 83 94 125 73 99 140 164 111 118 147 156 152 152 157 164 168 166 165 161 143 126 73 107 157 124 59 52 61 110 117 118 120 81 46 57 102 108 101 59 35 38 37 40 41 43 45 46 46 46 40 63 152 68 0 49 130 117 113 101 162 62 25 30 29 34 35 15 21 54 110 132 136 140 126 113 186 183 176 191 223 184 186 185 186 183 201 129 85 72 39 80 133 152 137 152 156 154 103 102 76 72 117 126 88 141 128 164 161 157 156 165 166 171 154 167 152 135 64 97 116 89 76 76 105 121 100 106 104 128 111 59 93 142 111 73 46 35 48 38 39 40 48 41 47 44 39 65 154 67 1 43 117 109 103 92 162 56 24 29 28 33 33 12 21 49 112 133 133 142 123 116 185 180 170 193 224 185 185 178 186 188 183 149 103 81 100 125 124 146 136 121 143 137 84 78 50 69 104 74 80 127 133 142 156 163 164 170 162 164 170 149 105 113 68 88 94 48 76 111 87 108 130 114 102 101 93 58 123 147 71 61 68 43 38 37 46 43 47 37 38 44 39 66 153 65 2 42 109 104 97 88 162 54 23 29 28 32 34 12 23 50 115 135 134 144 122 120 184 180 166 195 227 186 187 183 187 183 137 120 82 124 138 139 123 141 152 142 115 78 101 112 68 87 68 54 80 66 76 111 150 158 158 161 152 149 151 134 102 112 95 149 146 51 62 119 113 114 136 119 128 106 88 120 117 147 133 97 120 120 57 35 43 37 41 47 41 43 40 65 152 63 1 40 104 100 91 89 163 53 22 29 28 32 33 12 22 51 117 136 135 141 122 120 184 181 167 193 227 187 187 183 195 143 73 84 73 105 119 117 121 151 176 175 88 54 147 149 93 91 72 54 77 124 133 160 147 155 148 149 150 156 148 96 97 138 107 130 117 57 55 75 106 104 124 135 113 88 102 171 136 100 116 109 91 109 91 64 38 40 38 45 45 44 42 66 150 61 2 38 95 94 83 89 163 51 22 29 29 31 32 11 23 53 119 138 137 140 121 120 183 183 168 192 224 187 185 193 163 102 91 84 65 46 92 161 127 132 164 162 138 115 152 131 87 47 95 142 104 124 98 90 120 140 139 144 144 146 127 102 119 114 68 125 123 67 51 70 80 109 108 150 118 95 128 148 163 106 94 118 91 96 89 87 40 43 42 34 42 44 43 66 147 59 2 36 87 89 75 89 161 49 21 29 29 31 30 11 23 54 121 140 138 141 123 119 179 183 171 195 228 185 188 169 129 110 98 106 61 61 71 141 122 145 166 119 158 120 94 123 104 63 64 106 126 112 97 92 108 132 132 130 128 127 84 71 100 111 71 83 74 44 56 127 87 60 83 141 108 76 115 159 156 87 91 117 146 149 99 127 72 35 44 38 45 45 45 67 144 57 2 35 81 85 69 89 159 49 21 29 29 31 30 10 23 55 122 142 139 142 124 117 175 183 173 197 230 189 190 123 105 101 84 117 71 70 133 144 100 124 165 137 127 89 94 129 130 133 62 71 126 138 131 149 124 137 123 135 128 112 81 79 105 115 102 49 51 68 90 124 141 128 98 121 141 137 137 131 143 104 56 91 146 142 109 159 89 33 44 44 46 44 42 65 143 56 1 34 77 82 64 89 160 45 23 26 28 30 31 10 22 57 123 138 136 142 121 118 176 185 176 199 228 184 138 83 109 89 99 111 72 81 147 112 108 147 149 140 103 101 100 75 90 127 100 64 83 119 110 104 111 129 108 133 132 93 74 87 100 145 103 51 59 79 128 144 166 150 127 117 104 121 118 60 100 124 98 87 124 118 117 157 85 48 44 40 43 44 40 63 144 57 1 33 72 77 60 88 159 40 24 24 28 31 32 11 22 59 124 136 132 141 119 120 178 186 177 200 227 171 124 83 136 109 85 79 91 88 126 151 146 126 161 138 120 173 139 93 67 61 74 70 107 82 94 119 101 123 133 144 128 91 109 78 124 122 57 111 112 79 110 143 126 128 121 89 104 139 80 43 123 122 142 94 69 61 108 129 84 68 39 39 50 45 43 65 147 58 2 31 69 74 57 86 159 40 24 25 29 31 31 10 22 60 124 137 134 141 119 120 177 185 176 199 228 146 163 115 130 97 58 84 114 100 161 139 137 110 158 120 79 127 139 162 144 97 41 84 87 44 67 146 137 116 132 136 126 116 109 60 53 70 110 145 151 93 122 151 152 154 97 67 109 154 123 63 114 73 109 136 103 61 77 88 70 81 41 36 49 47 42 67 151 61 3 29 67 69 52 83 159 39 23 25 29 30 30 9 22 60 125 138 135 143 121 121 179 186 177 200 228 135 151 101 89 66 73 130 86 68 138 127 111 140 126 149 128 104 126 171 135 79 50 113 74 51 52 79 138 148 152 145 120 68 63 127 111 100 108 140 128 87 131 139 149 127 72 85 116 97 108 108 101 51 97 116 112 118 111 133 120 119 73 46 44 46 41 66 154 63 2 30 64 64 48 83 160 37 22 25 29 31 28 9 22 60 126 139 136 144 122 122 178 186 176 200 229 124 100 77 73 66 118 116 62 46 76 145 122 103 99 150 135 140 138 136 90 55 71 116 82 67 50 68 117 123 119 121 88 78 110 135 160 161 137 162 142 113 133 93 82 88 69 66 111 121 95 92 81 59 135 116 84 93 110 131 115 105 88 52 43 46 41 66 155 64 1 31 63 62 47 82 161 36 21 25 29 30 27 9 22 60 126 140 137 143 119 120 176 183 173 198 230 106 92 82 96 88 132 128 76 62 44 70 101 65 59 108 103 102 114 87 55 62 125 123 147 93 81 118 50 94 122 89 65 112 131 132 105 126 177 138 149 133 138 108 60 71 94 107 135 134 134 110 127 91 102 104 83 83 87 88 94 81 91 78 49 43 37 68 156 59 2 30 62 60 43 84 161 35 22 25 29 29 27 10 22 61 127 141 139 142 116 118 172 181 171 201 229 147 151 115 82 83 151 156 69 59 77 92 91 72 82 145 100 77 136 96 48 91 143 118 177 127 103 100 43 132 157 70 75 118 134 136 120 65 144 165 130 128 134 133 135 97 97 124 152 154 137 95 114 121 126 132 81 75 69 89 87 73 86 86 50 54 38 68 154 53 6 29 58 58 37 86 158 33 23 26 27 27 26 12 20 63 125 142 139 141 114 119 167 182 171 209 226 106 100 69 71 81 132 128 117 102 97 135 110 89 100 131 139 126 101 101 81 91 117 92 108 138 77 80 93 107 120 82 65 141 151 153 148 78 95 146 118 123 143 131 148 125 104 127 142 186 140 64 55 112 129 105 98 129 129 82 84 87 86 103 88 76 52 63 150 52 4 28 56 56 38 87 159 32 24 29 30 29 26 9 19 62 124 139 138 141 115 120 168 181 169 207 229 71 59 70 99 89 111 95 81 61 51 98 117 124 143 155 132 108 81 101 112 104 116 137 122 106 79 94 141 98 85 86 63 136 130 141 172 125 99 149 166 125 111 131 123 127 158 149 124 169 140 91 62 115 160 130 115 94 103 101 72 63 77 103 91 78 68 72 145 49 2 27 53 53 37 88 160 31 23 29 32 30 26 10 21 65 124 139 140 142 116 121 169 181 167 206 228 94 66 87 111 86 88 91 117 88 101 136 98 110 138 160 127 86 106 91 92 92 75 123 125 92 94 67 77 129 95 83 105 141 152 139 144 127 115 166 138 113 131 102 108 102 124 175 167 168 114 83 79 107 105 67 106 104 116 133 120 86 96 141 114 78 78 91 138 50 3 28 51 51 35 85 159 30 22 27 30 28 26 11 24 68 126 140 143 143 115 121 169 181 167 204 225 103 99 90 99 131 115 71 125 98 119 132 71 99 91 92 126 93 101 135 85 83 151 128 146 147 132 57 64 106 97 119 174 128 159 159 138 141 156 181 125 118 146 98 129 166 140 172 164 162 120 96 114 97 103 84 127 134 106 102 166 140 130 171 156 93 69 88 115 50 5 29 50 49 35 85 156 30 19 23 27 27 26 10 24 69 126 139 143 141 113 121 171 183 168 205 225 102 115 110 70 91 84 44 100 75 78 103 84 97 83 52 93 82 85 122 63 82 136 137 124 103 88 83 64 90 110 95 111 68 67 117 138 155 165 163 144 150 151 137 91 171 137 139 159 166 107 60 127 110 83 59 102 100 102 117 125 114 132 154 147 122 61 71 92 43 0 31 49 49 34 86 155 29 19 22 24 27 26 9 22 67 125 137 142 142 111 121 171 183 167 206 227 80 109 116 100 111 86 54 101 90 42 64 69 60 92 32 100 85 83 95 73 81 81 116 122 93 124 138 82 82 35 39 63 71 78 141 152 148 154 141 135 139 141 153 126 65 68 89 142 150 77 85 135 101 55 51 110 79 94 141 113 114 139 146 147 135 75 57 78 43 6 32 47 46 31 90 159 28 23 24 24 27 23 7 17 68 126 142 142 144 109 120 169 179 163 210 226 77 144 128 119 127 96 63 142 126 88 99 56 37 101 71 102 114 107 71 84 109 147 151 117 141 166 128 120 107 56 77 64 80 132 166 121 148 193 122 162 124 116 138 139 130 59 75 137 174 89 66 76 77 97 87 94 86 82 96 89 110 104 125 135 91 78 72 76 36 5 31 45 44 30 90 157 27 23 24 24 27 23 8 19 70 127 141 141 140 107 120 169 181 167 211 226 80 108 75 120 127 79 67 111 99 102 82 24 17 114 124 104 114 130 75 107 109 134 154 159 144 128 142 120 114 107 156 115 60 105 141 120 140 149 113 124 122 133 117 101 155 84 80 118 146 76 55 34 83 111 86 122 134 121 117 104 125 77 74 94 88 129 135 107 38 1 29 43 40 30 91 155 27 23 24 23 26 23 8 20 71 127 141 139 139 108 122 169 180 166 212 226 96 89 72 110 98 64 70 106 96 100 78 23 22 87 82 82 77 107 98 99 121 96 132 160 120 90 139 102 99 89 114 131 84 66 142 165 153 140 162 139 152 105 149 120 74 99 127 162 143 70 62 43 59 82 74 127 146 134 130 148 187 125 69 92 110 107 101 94 45 4 28 43 37 29 94 153 26 22 23 23 26 22 7 20 72 129 142 139 141 112 126 170 177 164 211 226 101 78 48 53 61 64 74 94 104 93 50 13 42 87 94 113 85 93 91 100 76 81 125 141 120 76 49 81 159 140 78 75 98 110 133 142 151 154 140 121 133 130 95 105 93 148 152 159 151 92 65 56 44 71 85 127 132 156 140 131 135 94 88 161 140 80 46 64 45 7 29 44 36 31 97 154 25 22 23 22 25 22 6 19 73 131 144 141 140 112 126 170 178 165 212 226 65 67 61 72 101 99 107 75 93 90 36 15 62 116 89 109 100 74 84 100 103 118 135 118 136 107 49 84 173 176 131 59 81 108 77 76 81 76 77 103 137 140 100 95 148 173 170 162 158 101 78 81 59 92 85 126 131 149 138 125 96 51 80 173 115 90 58 59 33 3 29 44 36 30 99 154 25 21 22 22 25 21 5 18 73 131 144 143 139 109 126 170 178 168 214 226 119 141 101 73 124 151 146 110 84 51 32 23 60 117 90 97 89 72 83 100 100 109 129 131 144 71 67 86 147 131 94 54 129 91 62 59 64 77 143 129 119 119 53 89 119 132 181 162 101 98 118 74 64 88 115 131 109 128 133 148 131 90 88 127 102 98 76 46 19 7 24 39 35 20 102 150 20 21 23 24 27 21 6 19 76 129 139 139 146 104 127 168 168 164 219 224 121 125 115 97 125 120 103 92 49 32 31 21 84 114 78 123 107 75 80 123 105 117 75 107 78 86 146 122 112 79 64 90 136 85 57 54 82 125 152 118 124 149 144 135 95 86 179 155 81 66 109 97 74 112 107 115 121 113 149 155 147 123 104 99 105 100 79 43 17 7 26 40 35 22 105 151 21 22 24 24 26 21 6 20 77 130 140 140 146 104 126 168 169 164 220 223 110 105 82 94 145 116 91 65 26 33 37 27 45 75 94 110 142 78 87 122 121 103 64 47 80 124 145 136 91 71 93 144 129 126 84 70 99 77 107 149 119 98 146 126 86 88 149 109 64 68 110 102 51 56 74 121 138 88 124 156 110 68 95 102 100 104 82 42 16 10 28 40 36 24 106 150 21 22 24 24 26 20 7 21 77 131 140 141 145 102 126 168 169 166 220 223
##       .src    ImageId     .rnorm .pos Image.pxl.1.dgt.1 .lcn
## 1908 Train Train#1908 -0.3524003 1908                 5  OOB
## 2788 Train Train#2788 -0.8952096 2788                 1  Fit
## 1862 Train Train#1862  0.6062188 1862                 1  OOB
## 6493 Train Train#6493  0.7944260 6493                 1  OOB
## 6406 Train Train#6406 -1.2134735 6406                 1  Fit
##      left_eye_center_x.All.X..rcv.glmnet
## 1908                                  NA
## 2788                            66.39388
## 1862                                  NA
## 6493                                  NA
## 6406                            66.73648
##      left_eye_center_x.All.X..rcv.glmnet.err
## 1908                                      NA
## 2788                                31.04543
## 1862                                      NA
## 6493                                      NA
## 6406                                28.78430
##      left_eye_center_x.All.X..rcv.glmnet.err.abs
## 1908                                          NA
## 2788                                    31.04543
## 1862                                          NA
## 6493                                          NA
## 6406                                    28.78430
##      left_eye_center_x.All.X..rcv.glmnet.is.acc
## 1908                                         NA
## 2788                                      FALSE
## 1862                                         NA
## 6493                                         NA
## 6406                                      FALSE
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
## 20 fit.data.training          9          1           1 122.385 132.313
## 21  predict.data.new         10          0           0 132.313      NA
##    elapsed
## 20   9.928
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
##      mouth_center_bottom_lip_y
## 7050                        NA
## 7051                        NA
## 7052                        NA
## 7053                        NA
## 7054                        NA
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Image
## 7050                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                182 183 182 182 180 180 176 169 156 137 124 103 79 62 54 56 58 48 49 45 39 37 42 43 52 61 78 93 104 107 114 115 117 122 120 122 118 114 115 118 117 123 122 122 112 113 118 125 124 122 109 101 96 102 108 107 100 89 76 67 58 48 44 36 32 21 12 5 15 21 24 33 32 41 52 63 71 90 106 121 152 173 180 183 182 182 183 182 182 181 182 182 183 182 182 182 182 181 181 181 182 179 177 162 136 106 84 64 49 44 40 50 50 42 36 33 32 43 51 67 77 94 106 121 124 128 127 126 128 128 129 126 126 122 128 123 128 123 129 127 133 123 128 129 135 127 122 111 115 118 120 121 118 114 107 107 99 89 74 70 59 47 29 18 16 22 25 25 34 39 45 51 52 60 84 97 124 157 175 181 183 182 183 182 184 181 183 183 183 182 182 182 182 181 180 183 182 180 176 152 111 77 59 45 36 40 51 48 37 29 29 33 45 61 81 95 107 113 120 126 129 128 132 132 130 132 132 132 131 133 130 132 129 130 133 137 136 137 135 137 137 134 129 125 126 131 131 128 128 124 124 122 121 111 105 98 86 72 62 42 27 20 19 14 15 27 40 36 27 41 56 77 107 138 163 176 180 183 182 182 183 183 182 183 182 182 181 182 180 181 180 181 180 180 171 141 106 78 54 34 34 41 41 31 28 31 39 50 72 88 108 115 123 126 132 131 134 135 137 137 136 135 136 137 134 133 134 137 136 143 140 142 137 138 138 142 138 138 132 129 126 134 135 133 134 134 134 128 128 120 119 116 112 100 92 74 55 37 30 20 6 3 13 19 26 42 48 62 81 105 139 167 179 182 183 183 181 182 182 183 183 182 183 182 181 182 182 180 181 166 135 98 68 46 29 24 28 34 29 28 28 34 43 74 91 113 118 129 126 137 134 140 136 140 135 138 142 142 141 135 136 134 137 131 142 138 144 138 142 135 141 136 141 137 138 131 126 129 132 134 134 136 133 136 130 131 122 125 118 118 107 103 90 79 58 44 24 10 10 15 20 28 41 46 63 80 123 166 180 182 182 181 183 182 183 181 182 182 183 182 182 182 181 182 178 142 94 66 47 35 26 24 30 28 24 21 24 39 69 94 115 124 131 132 137 139 136 133 139 137 136 140 143 145 143 141 138 140 137 139 136 140 139 140 138 139 139 144 143 142 141 137 134 135 135 135 135 132 135 133 134 129 130 130 128 124 122 117 117 110 92 65 44 29 21 18 15 19 27 32 43 60 86 130 169 181 182 182 182 183 181 182 182 184 182 182 180 182 182 182 172 129 85 55 36 29 23 20 23 20 20 17 26 53 93 114 128 127 138 136 139 137 137 132 135 135 139 142 146 146 147 150 144 145 140 142 135 139 135 140 140 144 140 147 143 145 136 139 140 140 140 138 136 137 140 137 138 132 133 129 131 127 129 121 126 118 107 87 70 46 21 18 15 15 27 34 40 51 60 92 145 177 182 182 181 181 182 182 182 181 181 181 182 181 182 183 166 111 69 43 26 22 19 20 21 19 13 20 36 79 112 132 134 142 137 143 138 142 135 139 136 140 142 147 146 147 147 146 149 143 145 136 141 137 141 137 142 139 149 145 148 140 140 139 140 136 135 140 140 141 136 139 134 140 134 138 131 135 125 128 121 124 112 104 85 62 31 16 18 15 22 37 39 35 50 82 127 165 180 182 183 181 182 182 182 182 182 182 182 182 182 183 146 85 58 37 21 16 16 14 12 14 19 32 61 103 131 139 143 141 141 138 138 135 136 137 137 138 141 143 142 143 144 144 144 143 140 140 138 141 140 144 143 146 144 147 147 146 143 144 144 136 137 139 141 141 139 135 138 138 135 132 133 133 133 129 129 124 123 118 106 82 50 22 20 26 19 26 36 38 44 64 109 160 178 182 181 182 181 183 182 183 181 182 180 181 182 178 126 66 42 25 13 12 16 12 14 17 26 47 89 118 139 141 145 138 140 139 140 131 135 132 134 133 139 140 140 140 145 145 144 145 141 144 139 143 141 145 139 146 142 147 142 145 142 145 138 133 132 136 142 141 142 139 141 134 136 131 134 128 133 129 133 128 128 121 116 101 74 40 24 27 24 22 34 45 48 57 85 144 178 182 181 182 182 183 181 181 182 182 181 181 183 176 114 53 28 15 10 13 14 12 17 26 39 68 95 125 137 147 140 141 134 138 130 135 130 136 134 139 136 140 139 137 141 144 144 141 145 138 141 138 143 138 143 142 149 145 145 142 145 144 142 137 139 139 135 138 137 140 135 138 132 135 128 136 131 131 124 131 126 125 117 114 92 64 33 27 28 24 27 35 44 41 57 117 171 182 183 181 183 181 183 181 183 182 182 182 184 158 79 29 18 14 17 17 11 15 23 32 52 76 100 121 140 148 143 136 138 136 135 133 133 134 132 129 129 131 136 136 139 140 139 140 141 139 136 136 137 139 139 142 142 141 141 141 140 140 141 143 146 144 141 140 140 136 135 131 132 131 129 129 131 130 127 127 131 127 121 114 101 73 46 28 23 27 28 27 24 29 47 85 149 178 182 184 182 182 182 183 182 182 179 182 178 133 63 30 19 19 20 15 11 19 29 33 55 79 104 121 139 143 146 136 143 136 136 128 131 127 131 128 132 132 133 130 133 136 134 135 137 141 138 141 135 139 134 139 136 140 138 143 138 141 141 141 139 143 143 137 139 135 135 132 136 132 130 126 131 125 131 123 128 125 126 120 115 89 62 32 14 23 33 36 25 18 40 79 133 174 183 183 182 182 182 181 181 181 181 181 184 143 60 26 17 15 17 14 15 24 29 41 62 87 103 131 143 146 137 136 131 138 134 138 135 141 135 142 138 141 140 144 146 144 144 137 142 138 139 135 143 138 142 136 138 135 141 137 141 136 140 142 142 140 137 138 134 135 131 129 125 129 121 125 118 125 122 126 119 124 124 128 121 103 66 35 20 16 25 39 40 24 14 47 126 176 183 182 183 182 182 182 182 182 181 183 182 135 55 23 16 17 17 15 17 24 36 49 68 90 114 138 147 142 134 134 138 141 137 132 131 130 130 133 134 135 136 142 144 148 147 144 143 143 140 139 139 140 140 138 139 135 135 136 140 138 140 141 141 134 132 133 134 131 130 127 129 131 127 123 120 118 119 118 119 124 126 129 129 115 84 46 25 20 17 26 41 37 23 32 102 166 182 183 182 182 182 182 181 182 180 182 185 135 52 28 24 22 17 14 21 26 38 55 79 102 129 138 148 138 139 134 141 131 131 129 137 133 143 138 139 137 141 143 144 146 143 145 143 144 140 143 139 145 140 143 138 142 134 136 137 141 136 137 138 137 138 138 140 138 142 138 142 138 142 132 134 127 129 121 122 121 128 128 134 121 100 63 32 21 12 19 31 42 45 31 61 147 181 182 182 181 182 182 182 181 182 182 185 130 42 30 31 25 24 22 25 32 42 64 90 118 134 145 142 140 133 136 131 141 138 140 137 143 141 143 138 137 139 144 145 143 145 141 146 143 146 141 145 140 145 141 142 137 140 136 141 140 143 140 140 141 141 144 141 141 139 140 132 134 130 132 125 128 124 129 123 124 123 131 133 131 108 81 51 32 20 19 20 23 39 48 45 115 174 182 182 182 183 181 182 181 182 183 179 112 48 37 30 25 22 28 35 38 49 76 105 130 141 143 137 134 131 133 134 136 130 128 126 125 124 129 127 124 128 136 139 140 140 144 146 149 148 144 139 139 142 143 144 142 141 140 142 142 140 140 143 145 143 140 143 144 143 142 137 133 130 127 123 119 118 121 124 128 129 129 133 134 121 96 67 39 28 28 27 18 19 28 49 109 170 182 181 182 182 183 181 181 181 183 178 120 49 38 33 25 26 32 30 40 65 90 118 138 147 139 137 132 135 126 123 112 114 108 104 98 94 88 94 100 103 107 111 114 122 137 143 149 146 147 134 141 140 149 144 145 138 144 141 144 140 144 143 143 143 143 148 146 148 141 138 132 139 135 136 129 133 125 126 121 124 129 137 134 131 109 77 46 29 32 34 26 20 23 26 84 163 183 182 182 182 182 182 182 181 182 182 118 41 38 38 33 36 37 37 51 73 102 128 147 148 145 137 137 129 117 99 97 87 81 69 55 48 53 57 62 71 85 94 93 104 124 142 140 145 145 147 140 148 146 150 142 145 144 149 147 146 146 145 142 143 145 144 143 134 135 131 132 128 134 129 133 131 134 132 128 122 127 134 136 131 116 86 51 28 29 36 29 15 16 35 112 174 182 182 182 182 181 182 181 182 184 179 106 42 40 41 41 46 47 48 64 84 117 145 151 149 142 131 117 105 93 84 73 58 51 41 30 31 40 49 45 47 71 92 97 109 115 124 131 139 142 146 147 148 147 148 148 149 142 143 146 148 148 149 142 141 140 137 131 121 112 105 99 101 109 112 118 118 120 124 126 124 125 127 132 136 128 103 63 26 21 35 36 19 15 37 117 177 183 182 181 181 182 181 181 181 183 177 112 47 42 43 47 45 43 51 72 108 138 151 151 144 127 111 97 86 68 62 55 53 43 45 40 37 42 55 51 44 55 78 98 106 112 114 128 135 143 138 144 138 148 146 153 147 148 142 150 150 152 148 142 134 129 118 111 104 81 74 59 59 62 77 84 94 96 104 110 116 119 130 132 133 130 112 76 39 17 22 33 31 31 53 127 176 181 182 182 181 181 182 182 182 183 179 122 49 42 47 44 46 48 65 99 134 151 154 154 140 113 90 83 73 75 80 96 102 110 105 99 79 63 58 57 52 54 63 79 93 96 109 122 137 139 143 138 143 141 150 149 151 146 146 145 149 151 150 139 126 110 99 98 82 77 70 55 49 50 44 46 48 61 77 95 101 109 122 133 135 135 121 87 51 28 22 26 32 33 68 148 180 180 182 182 183 180 182 181 181 185 170 97 49 49 48 46 53 62 78 121 152 157 157 148 125 104 92 86 89 103 119 128 124 124 123 126 118 106 89 80 69 66 65 66 80 93 102 117 127 134 138 133 139 144 151 148 145 141 144 144 144 141 141 132 124 110 94 85 82 80 64 57 55 46 33 24 25 35 52 72 78 86 101 121 133 138 133 102 60 33 22 24 37 48 82 151 181 183 181 181 181 182 181 181 180 184 167 91 44 47 53 56 59 69 105 146 157 154 153 139 119 106 110 112 126 131 138 130 130 121 123 114 111 110 105 97 96 80 73 75 81 96 99 111 119 128 126 133 135 143 143 146 129 130 135 140 135 134 128 126 120 109 97 88 85 71 59 51 46 37 33 25 34 40 47 50 55 61 80 103 121 132 139 116 72 40 26 28 41 60 85 149 180 181 182 182 181 181 181 181 181 184 171 90 37 44 52 56 63 88 133 156 159 157 139 123 117 108 118 135 137 142 137 133 122 122 118 116 109 110 104 102 101 96 90 91 98 104 107 108 117 119 130 135 140 137 141 132 116 109 116 120 129 128 123 120 110 95 87 82 76 69 55 56 57 53 61 78 91 104 104 91 73 60 64 85 105 127 140 128 83 41 29 31 46 68 94 152 181 181 182 181 182 181 182 181 182 184 178 110 49 45 52 59 76 112 144 158 161 151 126 120 132 134 132 139 140 136 132 134 126 127 122 120 113 109 104 101 96 98 102 105 108 110 111 109 114 122 132 134 133 131 134 119 98 96 109 117 125 124 122 113 109 100 94 89 81 74 72 78 81 87 98 108 110 115 119 123 126 108 79 74 87 110 134 144 110 52 27 34 54 79 122 166 180 182 181 181 181 182 182 181 180 183 183 122 49 44 56 75 96 132 153 161 156 143 130 121 129 138 142 138 143 129 128 125 125 115 108 93 93 87 88 86 89 89 93 100 103 109 114 116 113 118 123 130 122 125 121 115 89 92 106 124 124 116 113 113 109 100 96 96 94 87 88 85 91 92 103 100 107 111 122 122 131 135 126 96 81 93 123 145 138 82 45 51 65 87 133 173 182 181 181 182 180 182 181 182 181 182 184 130 47 40 61 76 106 137 154 157 149 139 130 128 132 137 137 138 125 113 104 101 98 104 102 103 97 91 84 86 84 81 83 88 95 97 104 102 106 109 120 127 131 127 128 107 87 89 103 119 127 117 111 111 103 96 95 99 89 84 80 85 85 94 103 111 109 115 116 118 120 130 136 130 98 84 115 142 147 116 69 58 72 96 138 175 183 181 182 180 181 180 181 180 182 181 187 143 43 36 61 73 110 144 154 155 147 134 125 124 130 131 125 115 102 96 101 105 100 103 102 103 98 95 91 93 90 88 86 88 87 88 91 97 102 108 119 126 133 128 120 103 94 96 109 120 124 118 110 104 100 96 90 85 85 90 92 90 90 98 101 109 115 126 128 132 130 126 120 125 124 100 101 131 145 138 88 53 57 87 143 175 181 182 182 180 181 181 180 181 179 181 187 152 57 45 58 77 116 150 158 156 144 130 123 124 124 113 108 104 108 102 96 91 93 87 92 92 94 87 87 79 79 82 89 96 96 90 82 89 94 102 109 124 128 130 118 126 128 123 114 126 121 117 108 100 86 74 83 94 102 94 90 85 89 90 100 101 112 114 124 125 135 142 144 121 115 115 106 122 142 143 110 75 72 92 148 180 182 181 180 181 180 181 180 181 180 181 186 161 63 39 59 84 130 152 154 152 145 128 126 125 105 105 109 108 100 92 83 86 83 87 88 93 87 87 82 79 73 68 72 77 89 92 87 76 81 86 98 113 126 127 134 140 140 132 136 134 124 112 99 79 69 88 103 104 92 85 80 84 85 93 104 112 111 110 108 110 115 127 142 146 127 116 106 119 143 144 128 91 78 106 155 181 182 180 182 181 181 181 181 180 181 179 186 157 59 34 50 84 135 156 155 151 144 138 129 98 104 122 107 92 81 78 77 79 83 93 87 86 83 85 85 84 81 72 59 56 66 82 99 94 77 75 85 100 125 136 134 122 120 126 142 141 126 103 84 68 88 101 94 88 78 75 72 75 81 92 99 106 105 103 100 102 100 109 123 139 143 136 104 101 129 149 143 105 68 95 161 180 181 182 182 181 181 181 180 181 180 180 185 166 74 28 40 93 137 138 137 130 118 104 86 92 133 123 97 78 70 69 93 93 85 74 61 48 55 56 63 73 82 82 66 47 48 64 93 119 102 78 75 92 88 83 80 87 88 100 89 103 109 93 64 92 114 90 81 73 63 61 73 75 83 83 93 89 93 96 98 93 97 93 103 118 141 145 133 105 98 138 149 116 77 103 165 183 181 182 181 182 180 182 181 181 181 179 184 174 76 28 43 103 135 56 42 99 114 74 54 119 134 102 81 69 61 89 102 63 32 45 80 93 60 35 28 37 57 63 66 56 45 51 73 93 94 77 79 42 43 79 99 103 103 97 85 53 53 71 73 106 105 86 71 54 56 70 75 84 84 83 86 89 88 93 96 100 92 83 86 105 127 149 135 78 69 92 126 129 92 120 172 182 182 181 181 180 182 180 181 180 182 180 184 177 93 32 42 102 106 33 89 155 154 117 90 123 119 92 78 73 72 67 40 29 60 112 135 103 63 39 48 54 51 59 57 57 56 55 67 82 84 85 52 40 90 113 124 125 128 121 117 96 48 36 76 83 83 78 51 41 62 76 70 69 56 43 41 51 52 62 88 114 96 74 82 99 112 151 148 64 48 76 126 132 40 38 141 181 182 182 181 180 181 181 180 181 179 181 183 180 105 32 36 67 99 151 155 149 140 109 97 134 132 114 108 89 64 45 49 77 110 130 93 53 66 31 34 63 73 109 88 44 45 70 82 95 94 72 34 62 93 116 122 135 128 133 118 107 63 28 64 88 92 83 59 53 61 67 73 62 43 34 35 44 56 42 45 77 95 77 73 88 114 147 162 77 32 96 157 168 71 36 139 181 181 181 180 181 179 182 180 182 181 180 181 184 121 50 34 67 132 143 140 138 138 104 99 142 137 119 102 89 78 83 90 95 107 127 81 35 42 19 4 49 93 120 134 82 55 62 96 93 51 39 39 58 97 120 133 137 133 131 131 110 67 30 42 71 97 88 57 47 63 97 101 79 51 51 98 89 112 127 66 33 60 76 73 85 109 144 164 76 18 86 147 161 91 85 166 183 183 180 180 180 182 180 180 180 182 180 181 186 149 60 19 53 108 120 128 128 129 108 95 115 122 111 106 105 114 116 103 89 86 111 88 44 41 34 33 52 61 83 88 76 74 88 98 46 20 50 34 58 112 129 137 134 140 134 135 123 87 35 32 27 61 78 47 47 78 108 75 43 27 15 49 54 81 129 108 48 34 63 79 91 104 133 157 84 35 97 151 164 99 93 168 182 181 181 182 180 181 179 181 181 180 181 179 185 168 61 14 88 140 144 136 135 130 119 90 104 130 129 132 120 117 109 102 88 86 85 92 84 81 73 76 77 84 82 75 81 87 95 66 10 29 47 26 79 122 143 137 142 138 143 131 140 111 44 20 30 31 67 71 62 74 96 81 54 33 21 28 35 60 102 95 75 74 73 82 90 104 124 147 101 69 104 137 155 96 99 171 183 180 181 180 181 180 181 180 181 180 181 180 186 174 63 34 118 153 152 152 148 144 135 115 103 132 138 117 112 108 102 90 87 82 79 78 80 87 89 87 79 76 77 77 77 78 87 84 60 61 55 55 97 136 142 143 141 142 142 138 142 129 70 35 43 44 63 78 75 70 77 79 72 63 64 51 47 81 92 87 98 112 110 103 103 110 134 157 138 114 124 141 132 67 112 176 181 182 180 181 180 181 180 180 180 182 169 146 135 140 59 43 125 152 150 157 152 150 143 135 114 114 129 117 105 99 100 95 96 93 89 82 82 76 77 76 77 77 78 70 75 87 99 100 98 79 65 87 116 140 143 147 138 139 135 137 131 133 97 64 55 75 86 76 65 70 71 73 81 83 88 84 83 84 83 83 100 113 128 130 126 128 143 153 141 133 140 156 144 114 152 181 180 180 180 181 180 181 180 181 179 170 147 128 116 101 99 88 128 151 155 153 155 148 145 137 132 114 104 118 112 109 103 103 97 104 100 102 97 97 91 90 84 83 76 78 83 104 113 97 83 79 84 104 133 137 149 144 143 135 141 129 128 124 113 83 70 68 90 96 78 70 71 73 72 74 71 75 70 75 82 98 107 120 126 134 135 142 145 134 127 137 146 153 162 156 172 184 182 180 182 181 182 180 181 180 181 141 89 105 116 126 123 102 128 153 156 152 150 149 144 140 137 130 111 102 112 120 116 106 104 104 104 102 98 96 93 93 90 89 88 94 107 120 100 61 67 92 107 127 134 138 139 141 138 133 133 129 122 118 115 96 80 56 50 96 101 77 75 78 83 84 88 86 89 90 97 103 111 114 122 127 134 133 125 124 134 144 150 152 160 153 138 153 175 182 181 180 179 179 181 181 179 142 59 66 92 118 110 105 132 154 153 156 151 145 138 140 137 132 132 124 111 103 109 110 116 109 111 105 106 102 104 99 98 94 105 112 112 97 67 57 76 98 116 133 131 136 132 137 134 138 132 134 123 118 111 105 88 71 44 67 99 93 82 80 88 89 95 96 102 100 106 105 114 114 120 122 120 112 116 129 135 141 150 154 163 153 117 120 146 172 181 181 179 180 180 180 180 153 79 66 95 103 104 105 139 150 156 153 155 145 143 142 141 138 137 138 132 119 101 97 96 102 100 101 99 103 97 100 96 101 103 102 103 111 108 102 108 115 128 128 129 125 133 130 138 134 140 134 134 122 115 110 112 100 97 104 113 109 93 87 90 93 84 97 102 107 102 108 108 114 109 110 112 121 123 130 136 143 151 157 162 161 135 118 119 157 182 180 181 180 180 179 181 157 85 80 118 108 89 108 138 149 154 156 155 154 153 149 144 141 139 135 135 133 125 117 109 100 98 96 97 93 92 98 104 112 116 123 123 123 123 128 129 132 132 127 121 124 127 134 139 139 136 135 131 127 121 112 119 118 109 109 117 121 116 104 95 90 92 94 98 101 101 103 105 107 113 123 125 127 130 129 136 146 150 156 161 163 136 89 93 159 182 181 181 181 181 180 179 157 86 95 126 122 95 109 136 154 152 154 155 161 158 153 147 148 145 139 138 129 128 123 126 120 122 115 119 115 121 115 128 128 129 128 131 131 134 137 136 135 126 122 119 133 136 144 138 142 131 134 129 137 130 124 118 124 122 120 118 119 126 123 118 105 105 101 105 104 109 106 116 117 123 120 127 127 129 132 139 141 146 151 158 168 126 59 100 170 182 180 179 181 180 181 179 161 94 93 142 132 113 109 139 145 150 149 158 159 159 151 146 151 150 146 141 140 132 133 128 132 125 129 127 132 127 132 130 134 132 135 138 137 134 132 133 130 127 120 133 139 146 142 145 139 140 135 138 138 143 134 128 127 130 131 125 126 130 133 129 128 114 112 105 109 104 111 112 121 117 122 123 132 134 137 140 143 147 150 155 162 128 71 113 173 182 181 181 179 181 180 180 166 103 102 143 144 123 111 132 145 148 153 156 156 155 154 153 154 149 145 147 146 142 141 142 139 140 139 136 136 137 140 138 139 137 140 143 142 135 132 129 125 116 122 140 146 143 149 147 141 141 138 139 140 142 142 138 127 125 127 127 132 136 136 135 138 138 130 122 119 115 116 118 122 123 124 127 134 137 142 145 148 151 153 157 161 127 80 112 169 181 180 180 181 179 180 179 171 116 120 131 123 81 88 125 144 147 152 151 154 153 156 154 153 149 147 152 149 150 145 145 139 150 146 149 141 144 138 144 140 142 139 142 138 131 127 125 119 112 123 141 149 141 152 150 151 141 142 143 149 141 148 150 135 119 123 131 132 136 136 140 139 144 137 142 132 132 123 130 126 132 129 138 138 143 146 151 151 153 153 156 162 130 81 115 172 182 181 179 180 180 180 179 178 134 118 123 89 62 78 128 138 148 151 155 151 154 151 150 151 151 154 155 154 149 152 145 148 148 155 147 147 141 148 146 143 140 142 139 133 120 114 125 121 121 124 132 131 139 142 156 148 146 143 152 151 150 143 147 132 113 119 127 132 133 138 138 143 140 143 140 144 138 141 136 142 136 137 137 145 147 149 152 155 155 155 155 156 133 92 138 180 181 180 181 179 180 179 181 180 152 121 112 90 59 69 120 137 142 148 151 149 149 148 147 148 151 152 152 154 152 151 151 153 154 157 157 152 148 150 146 145 138 138 133 122 109 115 126 131 130 126 116 111 113 126 142 145 143 144 151 150 138 123 122 123 112 105 115 125 135 139 139 140 139 144 145 147 146 146 143 144 143 144 142 145 145 149 152 152 156 154 154 154 127 110 156 182 181 179 180 181 181 180 179 183 171 132 102 91 63 71 106 134 136 145 144 144 142 146 147 150 150 148 152 155 157 152 155 151 156 153 158 149 152 146 149 141 142 133 128 112 98 118 138 145 137 123 109 106 102 115 121 136 138 137 134 132 115 108 107 119 122 110 108 120 134 137 143 142 145 146 151 142 148 145 150 143 149 146 147 143 148 150 152 153 154 150 152 150 119 122 168 181 181 182 179 181 180 179 177 182 177 139 98 81 72 64 106 129 137 142 145 143 142 142 144 146 149 153 154 156 153 156 151 157 153 157 153 153 147 149 146 142 132 130 122 100 97 130 148 142 128 112 98 92 107 108 119 122 131 123 118 109 102 98 97 107 135 126 101 114 129 138 139 144 143 151 145 149 146 153 149 152 150 151 144 147 150 151 151 155 154 149 147 140 117 141 177 181 180 179 180 178 180 179 180 182 181 140 99 82 73 63 95 125 131 136 143 141 143 143 144 145 149 154 153 151 152 153 152 153 154 155 151 152 150 150 144 138 133 129 115 100 102 127 130 123 105 65 30 23 63 101 108 110 111 104 100 98 97 77 66 90 125 133 110 108 125 134 140 143 144 147 149 152 152 151 148 149 150 149 149 150 149 151 150 152 147 145 145 134 116 154 180 181 180 179 180 179 179 179 179 181 181 146 102 87 71 63 84 126 125 134 134 137 138 141 142 147 150 150 150 149 153 149 152 148 153 150 156 148 150 142 143 136 133 121 111 97 100 110 115 106 61 6 0 0 8 76 100 97 91 90 91 97 62 15 12 50 99 120 110 102 117 130 139 140 147 143 150 146 153 149 151 146 152 147 151 148 150 146 147 145 141 138 140 126 120 166 181 180 179 181 179 180 178 179 178 181 181 152 117 98 78 50 70 110 128 128 129 131 136 140 144 144 147 147 145 149 149 153 146 147 145 152 146 145 137 140 133 133 127 122 112 97 88 98 103 88 40 3 0 6 22 53 93 81 74 79 91 54 2 0 6 31 79 108 101 101 116 131 134 140 140 145 143 149 148 153 149 151 147 147 143 146 147 149 144 139 138 135 135 117 122 169 181 179 180 179 180 178 179 178 180 180 183 155 114 113 100 60 61 100 122 127 126 127 132 132 137 139 142 142 142 144 143 144 143 144 144 145 141 142 137 137 133 129 124 126 112 100 86 81 76 75 69 50 43 66 65 57 77 73 68 72 68 49 25 21 38 61 81 88 96 107 120 126 134 137 138 143 144 147 145 149 147 148 145 144 140 142 143 144 140 138 135 132 135 114 128 172 180 180 180 178 179 180 179 178 179 181 184 167 114 101 111 119 89 100 120 124 123 128 128 130 131 137 142 141 138 136 142 138 143 138 144 141 144 137 140 132 134 126 128 124 119 110 102 87 78 79 92 112 119 116 92 79 72 70 65 67 70 98 92 87 83 78 69 81 100 115 124 128 134 135 142 138 144 138 142 142 149 142 146 143 142 138 145 134 132 136 133 131 128 111 147 178 178 180 179 180 178 179 178 180 179 180 182 182 142 95 116 147 105 77 112 119 126 127 129 124 126 132 134 139 139 141 138 142 141 141 137 140 136 139 136 140 133 132 130 126 119 119 118 112 101 88 91 99 111 106 98 83 72 67 70 71 87 98 101 95 87 77 79 99 115 122 128 131 130 135 137 143 139 138 139 144 140 146 143 143 141 140 136 133 130 134 130 133 124 111 155 180 179 178 180 179 180 177 180 179 179 179 181 185 168 111 115 147 107 71 104 115 124 123 124 120 121 126 132 132 135 138 134 138 139 139 134 134 132 138 137 138 127 130 131 130 125 126 123 120 113 108 104 106 104 97 97 99 93 86 87 87 91 90 91 86 92 96 109 116 122 125 131 132 132 131 134 139 138 138 139 143 139 142 142 145 140 138 130 128 126 126 122 125 118 120 164 180 181 179 181 179 180 179 179 178 178 182 181 182 181 140 115 147 110 78 99 112 113 121 118 120 117 123 127 128 133 133 136 133 140 135 137 129 132 128 133 128 132 129 136 134 132 126 123 121 113 96 87 94 92 89 89 102 107 117 112 97 75 74 76 82 94 106 108 120 125 128 131 131 133 130 134 132 136 134 141 136 139 135 143 141 143 134 131 125 127 126 121 122 112 131 173 182 180 180 178 180 178 179 177 179 177 180 180 181 183 171 137 140 120 79 89 104 113 114 120 121 122 124 125 125 128 136 133 134 130 132 134 135 128 131 127 131 131 135 135 134 131 128 114 107 108 90 78 77 82 81 84 90 107 121 119 91 76 89 97 82 78 94 109 122 125 128 131 131 130 133 130 132 129 132 132 135 133 138 136 139 135 132 128 129 127 126 123 115 117 156 180 180 180 179 179 179 179 177 179 178 178 180 181 181 182 182 158 137 136 81 67 97 109 115 126 123 122 122 122 126 127 132 128 130 130 132 129 130 128 134 126 127 127 138 136 130 124 128 115 98 91 89 86 89 84 79 81 89 106 112 111 104 97 100 103 85 80 90 109 117 117 122 129 131 133 136 132 131 129 133 132 133 132 137 133 134 134 131 127 133 129 124 119 108 126 170 181 179 180 178 180 179 178 178 179 178 178 181 180 180 180 184 169 140 141 103 63 93 107 120 120 122 120 124 124 125 128 126 128 128 132 123 121 118 127 124 127 120 124 124 130 124 126 119 112 98 89 92 95 99 90 83 80 98 108 115 102 106 100 95 90 98 105 98 102 115 113 111 119 123 131 132 131 127 130 125 134 132 137 129 132 132 134 129 130 130 126 115 115 101 131 177 180 181 179 179 178 180 178 179 177 179 177 180 180 181 180 182 179 152 135 98 71 92 107 114 113 114 121 120 122 127 128 128 126 128 123 116 112 120 122 119 115 118 117 113 112 118 115 110 99 90 88 95 104 96 90 93 101 103 114 118 118 107 106 98 94 94 95 96 107 117 106 100 109 118 121 124 125 123 119 122 125 130 129 131 129 132 132 128 124 126 123 116 107 103 139 178 181 180 181 178 179 179 179 178 178 178 178 180 180 179 181 181 184 169 130 82 68 97 108 110 111 114 117 119 123 123 124 126 126 125 120 116 115 123 116 106 99 107 108 104 95 95 95 91 84 93 101 102 106 100 99 105 112 111 115 116 121 116 120 115 110 94 88 96 105 105 98 89 92 105 113 109 104 116 124 124 119 127 125 128 131 134 128 126 120 119 118 120 100 95 141 177 180 178 180 178 179 178 178 178 178 178 178 180 181 180 181 181 181 183 160 97 59 91 106 114 103 101 110 119 122 123 125 124 126 119 119 111 118 115 97 82 96 96 94 91 91 84 87 78 76 95 109 111 113 120 113 117 118 124 115 116 116 129 130 134 123 122 110 110 103 90 85 86 84 82 90 97 99 101 120 118 119 114 119 118 132 130 130 126 121 110 107 101 90 100 155 180 179 180 178 179 179 180 178 178 178 178 178 179 179 181 179 180 180 182 187 171 121 94 94 94 93 95 107 116 117 120 122 123 120 117 113 110 107 106 94 86 83 87 79 76 79 82 75 80 95 101 113 122 119 105 97 96 106 112 120 116 118 116 107 96 98 108 117 122 118 102 93 91 86 85 76 78 85 94 105 117 115 110 112 123 127 131 127 124 112 103 100 91 82 131 178 180 179 179 179 180 178 178 179 177 179 177 179 178 180 179 180 179 180 180 180 190 168 83 65 89 103 106 105 110 113 113 117 121 118 115 106 108 107 101 92 92 85 79 66 68 73 76 75 90 109 114 109 100 87 79 75 73 76 80 91 94 94 82 81 77 83 86 93 99 116 117 112 101 85 76 71 64 68 84 94 109 107 108 112 125 123 126 119 118 111 104 99 88 69 123 176 180 181 179 180 178 180 178 179 177 178 177 177 180 179 179 180 180 181 181 179 184 170 81 59 93 103 105 95 103 115 119 117 115 112 104 105 107 106 89 87 88 87 72 66 68 70 68 87 103 103 87 73 68 65 67 68 70 65 69 62 65 62 65 65 70 67 71 67 71 83 100 107 103 85 70 65 62 68 75 83 90 107 106 117 119 125 122 117 111 108 92 92 80 67 136 179 181 180 179 178 179 178 179 178 179 178 179 176 179 179 180 179 179 180 181 180 182 182 113 57 83 100 99 89 103 113 115 110 102 107 109 105 107 95 79 77 79 72 69 67 62 64 75 87 87 71 55 58 56 55 50 51 51 51 47 50 51 49 47 55 55 45 41 39 45 56 67 83 96 91 78 69 75 74 68 62 75 97 109 114 117 121 123 114 110 108 93 84 73 79 154 181 179 178 179 177 179 178 178 179 178 177 177 178 177 180 179 180 179 179 180 181 180 187 146 72 89 110 103 101 106 107 108 104 104 108 107 106 99 80 72 63 63 63 64 58 53 61 69 57 47 38 37 37 40 35 29 55 79 77 78 71 69 87 93 97 88 61 38 35 39 48 43 49 71 86 82 75 77 78 72 66 71 87 106 107 114 119 122 110 102 98 98 85 57 94 168 181 178 180 179 179 178 179 178 178 176 178 176 177 179 179 179 179 181 179 180 181 180 185 164 87 80 103 101 98 96 105 108 106 106 107 105 108 94 79 63 55 56 63 58 40 25 25 22 18 16 14 7 1 14 15 10 42 67 70 77 57 58 89 96 88 75 44 28 25 15 20 25 27 37 57 69 72 71 71 74 75 73 96 107 112 112 118 113 109 92 91 92 82 56 119 178 179 179 179 180 179 179 177 178 177 178 176 177 176 180 179 180 180 180 179 181 180 179 183 180 110 71 86 90 85 84 94 101 103 107 108 105 99 94 83 65 56 63 66 45 9 0 0 0 0 9 10 12 42 40 28 35 30 24 19 16 11 11 21 27 23 19 14 11 30 27 6 7 8 8 14 28 46 65 72 72 69 74 88 105 110 114 111 105 98 93 91 85 68 78 144 180 180 180 178 179 178 178 176 177 178 177 176 176 178 178 179 178 180 179 181 179 179 179 181 185 144 76 83 94 81 83 82 88 96 102 102 103 97 93 84 72 61 64 58 40 25 33 31 22 12 30 28 34 92 101 71 85 101 95 104 116 110 109 120 122 104 98 98 77 87 79 46 21 5 1 1 2 19 58 73 76 73 78 84 103 107 113 104 104 95 93 95 85 52 88 162 181 179 177 179 177 179 178 179 177 178 176 178 176 178 179 178 179 179 180 180 180 180 180 180 183 174 107 75 86 81 81 83 84 87 93 101 97 92 88 87 74 68 63 63 63 68 58 57 57 56 47 34 44 77 72 57 86 99 97 122 142 117 122 145 142 118 112 101 85 87 85 58 35 19 30 43 38 37 53 72 79 80 82 92 98 111 105 105 100 98 91 92 75 68 122 175 180 179 178 178 178 177 178 177 179 177 178 176 178 175 179 177 179 178 180 179 179 180 180 179 181 183 123 59 67 78 82 88 87 89 92 92 95 95 92 86 79 67 60 65 79 88 81 70 65 67 64 70 73 66 58 61 72 76 71 86 101 93 98 112 103 90 86 78 64 66 66 54 53 57 62 62 66 68 78 80 79 85 88 84 90 101 104 100 104 96 86 84 65 71 148 181 180 178 179 177 179 178 178 177 177 176 177 177 177 176 177 178 178 179 178 179 178 180 180 180 179 186 151 75 66 73 73 77 81 91 95 91 97 93 86 80 78 72 71 74 91 94 95 82 74 72 79 86 91 89 88 82 85 84 72 73 75 85 79 83 73 76 62 64 62 76 83 90 85 80 70 70 73 81 87 78 76 82 89 81 87 96 103 99 102 88 86 79 57 83 160 180 177 178 176 178 178 177 175 177 176 177 175 177 174 176 179 179 178 179 179 178 180 178 180 180 180 182 182 117 55 67 73 69 75 85 86 91 92 90 80 78 80 86 82 94 95 103 103 102 90 85 79 86 90 89 95 102 105 97 90 86 103 106 106 93 94 88 95 88 89 90 97 89 81 75 77 86 90 88 83 82 79 83 90 96 97 105 98 97 94 88 80 69 55 111 173 179 178 178 178 179 177 177 177 176 176 175 177 175 176 174 179 177 179 178 178 179 180 180 178 179 178 179 188 154 72 59 66 75 77 73 73 81 85 86 86 78 75 77 87 99 103 103 107 113 105 100 91 87 80 82 83 93 93 94 96 103 104 108 105 101 100 97 102 97 88 84 83 80 81 84 91 96 103 99 87 83 85 85 90 92 92 94 90 87 82 81 78 62 81 150 179 178 179 178 179 177 178 176 177 175 177 177 177 176 175 175 177 178 178 179 177 179 178 180 178 180 177 185 160 105 79 58 63 73 75 68 67 76 81 76 81 69 59 63 79 91 108 104 107 109 117 113 110 97 91 85 83 81 83 84 86 93 94 100 95 95 90 92 83 89 78 77 71 74 77 90 100 106 104 103 96 87 85 77 78 84 83 77 81 77 76 72 66 65 127 175 178 179 178 178 177 178 178 177 174 177 176 176 175 176 173 176 180 178 177 178 179 179 179 178 180 178 188 150 40 58 104 65 64 65 65 58 51 62 69 64 60 60 51 60 74 94 102 112 105 109 111 117 112 105 97 89 90 86 88 86 90 86 91 87 91 83 87 83 80 73 76 74 82 84 89 95 105 107 105 103 100 101 96 82 67 76 73 70 73 77 67 63 42 75 158 180 178 178 178 178 177 177 177 178 177 177 177 175 176 173 175 174 178 177 178 178 179 178 179 178 177 186 157 43 0 68 131 96 68 68 68 64 65 67 62 63 55 56 65 71 75 89 103 112 107 108 112 115 111 107 99 95 90 89 83 86 86 89 89 95 93 88 88 85 81 79 80 78 86 92 93 100 103 97 103 105 104 104 90 72 64 67 63 61 70 71 67 58 42 102 171 180 179 176 177 176 177 176 177 176 177 176 175 175 176 175 175 173 177 178 178 178 178 179 178 177 184 165 55 0 2 61 137 119 80 58 65 71 69 68 69 67 65 56 57 61 69 75 90 100 117 114 117 109 109 102 94 84 83 78 71 67 64 62 60 67 67 75 73 75 75 78 75 86 89 95 94 100 105 105 107 104 103 95 77 61 62 59 57 59 71 68 64 49 88 156 179 179 177 176 176 178 178 177 175 176 175 176 174 176 173 177 174 177 179 179 178 179 178 178 174 182 173 68 0 2 0 57 134 133 104 56 52 75 70 67 73 76 69 59 56 66 68 77 82 98 108 118 109 111 109 105 92 81 74 68 68 68 60 43 38 36 41 45 49 51 62 62 72 77 87 91 94 96 107 108 110 109 105 97 79 66 59 54 54 66 67 63 50 54 117 173 179 179 177 178 177 177 176 177 176 176 176 175 176 174 175 173 175 174 178 178 179 178 178 176 187 191 106 6 1 2 0 57 131 130 124 100 60 70 78 67 74 78 72 66 61 56 52 63 73 82 97 112 116 117 116 113 109 105 95 84 82 85 74 59 54 53 57 61 63 68 69 66 79 91 97 99 103 104 110 113 113 110 104 94 82 66 59 49 59 69 66 54 43 96 160 178 178 177 177 176 177 177 177 176 177 175 176 175 175 173 174 173 173 173 176 177 176 178 183 176 128 72 15 0 2 1 0 46 125 126 119 117 81 53 64 76 81 82 78 72 66 53 63 69 83 89 103 109 123 116 119 115 119 118 117 110 107 105 98 91 91 91 88 91 89 99 98 100 102 110 108 109 107 114 114 112 109 107 99 87 77 64 60 56 56 67 66 32 75 152 176 178 177 177 177 177 176 177 176 176 177 177 175 176 174 175 172 175 172 176 175 176 181 184 131 46 22 8 0 2 1 1 0 42 119 129 117 113 104 75 59 68 72 72 71 73 62 56 67 82 82 97 101 109 116 123 123 126 123 123 119 116 119 113 111 108 109 107 113 111 112 112 115 113 111 109 118 116 114 110 111 108 106 103 100 93 79 71 61 55 49 57 53 68 143 186 182 178 176 178 176 178 177 176 175 176 175 176 176 175 175 174 174 173 174 172 188 183 155 82 17 28 44 13 1 2 1 2 0 36 118 130 118 113 106 94 71 57 62 68 63 56 54 62 66 73 74 93 102 104 110 120 123 125 125 123 121 120 122 125 126 125 122 121 126 124 125 122 123 122 124 123 124 121 119 113 113 109 108 103 93 86 79 68 64 54 54 56 37 61 126 137 154 177 178 177 176 174 176 175 176 174 176 174 174 173 174 175 174 172 173 173 93 72 32 14 38 41 16 0 1 1 1 1 0 33 113 128 117 110 96 96 89 61 48 56 52 49 59 60 64 65 81 82 96 94 103 118 121 119 123 120 122 123 122 128 129 132 132 135 127 134 132 138 130 134 131 130 123 126 120 124 120 113 98 91 87 86 77 68 65 46 49 61 53 26 36 33 29 101 169 177 179 181 177 176 178 177 166 167 175 179 174 175 171 172 170 174 17 20 34 39 19 7 5 1 1 1 1 1 0 42 115 126 115 108 101 91 84 75 53 36 37 45 49 53 54 59 65 75 86 101 110 120 118 123 124 126 128 127 125 127 132 133 135 131 136 129 128 128 135 136 136 124 125 126 124 122 121 102 85 78 83 80 64 66 61 39 46 105 94 18 6 34 31 21 44 46 62 123 162 171 163 113 56 46 80 127 159 173 177 179 180 175 42 36 20 9 1 2 2 1 1 1 1 1 3 43 110 124 113 108 104 95 87 80 69 44 27 28 30 39 47 52 62 76 84 101 113 117 118 124 124 125 125 128 130 131 128 124 125 126 127 120 125 130 134 129 124 120 125 123 122 117 110 87 76 69 64 59 51 51 40 26 67 127 85 8 0 2 17 34 40 28 11 21 59 75 61 47 41 16 2 17 61 102 119 126 136 153 31 8 0 0 1 2 1 1 1 1 2 5 15 55 102 118 111 105 100 95 89 86 83 73 45 25 23 24 39 49 58 66 74 72 93 105 108 113 125 125 124 122 120 123 118 112 117 125 117 121 117 131 133 130 116 120 115 119 115 114 98 86 72 60 53 49 44 33 15 29 89 131 82 15 7 2 0 5 25 30 19 24 37 42 37 31 25 15 25 45 41 25 17 9 11 33 3 0 1 1 6 7 1 1 1 2 2 11 28 56 99 114 115 106 98 95 89 87 87 84 66 36 21 17 25 34 39 51 60 67 74 81 83 90 100 107 117 118 115 109 103 98 113 118 123 118 115 119 121 120 122 111 104 106 108 105 90 79 75 59 47 38 21 8 25 76 107 124 61 10 19 7 1 0 2 10 14 16 24 29 24 16 6 14 28 28 24 18 12 8 5 5
## 7051                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   76 87 81 72 65 59 64 76 69 42 31 38 49 58 58 47 37 33 32 33 35 50 55 54 50 51 61 78 92 100 101 79 55 47 52 50 47 39 38 52 46 25 25 39 52 63 59 67 79 68 48 44 44 43 44 45 53 56 55 50 48 54 58 51 52 67 72 86 76 64 75 97 97 83 85 81 67 53 45 39 33 33 53 73 60 44 47 72 98 121 120 117 127 120 115 117 68 85 84 72 63 52 57 69 69 49 34 40 46 57 63 58 45 39 30 31 35 47 52 47 52 53 54 70 84 93 104 94 67 50 48 49 48 39 35 48 53 44 33 35 47 66 73 73 73 69 58 48 42 41 51 52 54 58 51 49 53 60 59 49 46 61 72 87 82 66 71 93 108 95 83 85 81 64 58 54 42 33 48 66 57 45 53 72 88 110 125 125 134 132 111 106 63 78 84 76 64 52 53 58 56 47 37 41 48 57 64 66 56 44 38 28 32 41 49 42 46 56 56 64 75 83 91 97 82 60 49 50 47 42 38 43 56 60 52 41 48 62 73 72 63 61 64 56 44 44 45 50 50 53 59 63 64 62 60 51 51 60 74 87 85 60 65 86 104 105 88 86 86 74 64 65 56 49 58 68 66 55 57 74 88 98 116 127 139 133 110 106 65 73 80 75 65 59 54 53 48 41 36 39 50 58 69 72 61 49 46 41 37 39 42 42 42 51 58 61 69 75 75 87 91 69 52 53 50 42 40 44 50 56 62 64 68 71 70 69 59 50 58 65 61 57 50 50 58 69 74 68 69 66 63 59 53 54 73 92 84 61 60 82 102 97 92 87 83 78 67 65 67 66 70 73 68 62 63 79 92 95 108 126 142 138 122 115 70 71 64 61 63 60 54 50 47 39 36 39 53 60 66 69 59 50 47 53 48 48 46 45 46 48 55 61 65 65 67 72 83 74 58 56 57 54 49 43 43 49 51 55 56 63 65 65 62 52 53 63 70 64 63 62 69 70 69 66 70 72 70 66 50 51 75 92 80 67 66 83 102 87 78 87 89 77 71 71 70 57 55 66 72 67 66 88 95 95 106 124 147 147 110 107 66 69 54 44 49 56 58 52 45 43 40 38 44 55 62 63 56 48 47 49 51 52 56 54 50 50 54 59 64 62 57 60 70 76 63 56 56 58 62 59 53 51 52 51 55 57 58 60 61 53 53 59 62 60 61 67 66 66 67 73 76 73 67 59 56 65 80 83 76 67 65 81 99 83 68 82 92 86 82 81 69 61 63 65 67 70 80 98 96 91 105 121 142 147 107 96 59 67 52 36 30 37 53 57 47 45 47 50 51 53 54 53 51 50 51 45 45 43 48 51 54 54 53 57 59 60 55 54 61 74 71 59 57 58 62 69 72 68 68 67 67 61 57 62 62 52 48 54 54 55 53 60 64 71 77 80 76 66 66 63 63 75 84 79 73 67 65 76 94 85 67 73 85 91 85 82 74 63 68 66 62 77 100 100 85 91 103 112 131 145 136 109 50 56 48 38 29 22 34 49 47 46 59 64 57 50 44 46 48 47 52 52 38 42 45 49 47 46 50 54 55 53 57 56 62 70 76 68 68 68 67 67 71 76 75 79 76 75 68 64 59 49 42 45 46 44 55 66 74 74 75 72 70 72 68 64 68 78 82 82 71 66 69 78 84 79 70 68 75 81 81 82 81 73 72 70 81 98 99 88 86 102 95 101 127 137 136 122 49 53 48 40 35 28 31 41 45 48 73 79 63 53 40 40 45 43 43 52 43 35 45 54 49 45 49 54 60 58 62 65 66 69 73 71 69 71 76 75 70 73 73 80 84 77 71 68 65 59 52 48 46 42 55 73 88 90 82 77 74 68 56 59 70 77 82 82 71 67 73 78 81 77 68 64 64 70 77 77 78 77 80 84 92 96 99 104 106 101 89 100 127 137 131 128 46 55 47 38 38 36 30 40 48 45 75 93 75 57 40 39 40 39 41 49 53 40 40 47 58 60 64 70 70 78 81 84 83 83 86 88 75 73 80 84 83 79 82 80 79 80 86 83 80 78 64 63 61 58 54 62 76 92 91 81 70 65 70 72 75 79 84 76 69 67 73 78 78 72 65 65 64 73 76 74 74 80 82 89 98 104 116 111 92 85 88 96 117 134 130 128 46 57 49 38 45 43 29 45 61 45 73 102 80 60 47 41 40 38 43 53 56 49 46 48 55 65 69 81 89 95 104 105 105 102 98 97 96 87 86 87 89 89 87 82 84 90 88 85 88 86 79 78 79 75 69 67 74 83 90 88 80 77 76 78 80 83 81 73 66 66 70 76 76 69 67 65 62 71 79 78 77 74 81 93 102 108 103 86 72 76 84 95 111 127 130 121 46 53 48 37 42 46 36 41 60 52 76 104 77 56 50 50 48 41 47 57 58 53 51 52 51 62 76 87 100 104 111 113 118 117 115 111 111 109 102 95 92 95 90 93 96 101 98 94 96 99 101 99 98 89 87 85 85 82 87 89 82 76 78 82 81 89 79 69 61 63 73 82 76 73 73 62 62 76 81 72 69 69 75 84 91 90 80 72 68 76 80 89 108 122 123 123 46 49 48 36 44 55 43 40 57 56 72 101 76 53 49 51 52 48 51 56 63 57 54 54 57 66 82 92 98 105 109 115 120 127 129 128 121 120 117 113 109 106 110 111 112 113 116 105 107 114 115 114 109 105 98 100 99 97 89 87 84 80 82 82 83 84 76 66 63 67 81 83 77 83 76 58 60 77 78 77 75 74 75 81 83 85 80 65 69 81 81 80 97 113 115 119 40 44 51 38 40 56 49 44 56 58 70 90 72 53 49 50 59 62 59 58 64 59 55 56 60 79 92 98 100 107 107 111 115 123 127 128 130 126 125 124 123 118 120 123 126 123 122 116 113 119 118 118 116 111 105 107 108 106 94 83 79 76 75 75 79 74 72 65 66 71 81 86 84 75 62 71 73 71 74 81 78 75 75 77 80 79 69 61 65 66 62 61 64 86 107 109 35 43 51 39 42 58 53 41 55 64 69 75 67 53 47 54 66 66 59 65 71 69 54 55 65 84 95 98 102 99 103 108 118 121 128 128 130 131 131 131 127 127 126 130 126 130 127 121 115 121 128 127 125 117 111 101 97 97 95 85 80 79 78 79 72 68 63 65 66 72 81 82 75 75 85 89 86 78 82 85 85 81 81 82 79 74 65 59 56 53 54 61 57 58 84 103 34 35 40 38 48 57 50 42 51 61 66 69 62 53 51 61 70 69 66 71 72 60 46 57 76 88 98 103 104 99 104 112 123 126 126 130 132 136 134 134 131 132 131 130 127 132 138 132 123 121 127 128 126 115 107 98 95 93 91 86 86 82 79 80 75 71 67 66 74 82 83 80 87 93 95 92 90 90 93 93 90 87 88 87 78 78 75 61 52 53 49 50 59 59 64 76 34 26 31 36 46 55 47 38 46 56 59 67 62 53 53 62 76 78 78 63 49 44 57 72 88 95 103 112 107 105 103 113 121 129 130 132 127 129 132 126 131 138 139 129 123 123 130 128 122 117 116 118 113 113 112 115 111 107 103 102 94 85 81 72 71 73 72 70 80 86 88 86 92 99 97 99 94 96 94 95 92 90 90 88 85 82 78 70 65 60 47 37 47 57 61 63 33 24 25 29 39 49 44 34 45 55 59 61 57 49 55 69 81 86 83 72 61 65 76 86 91 94 97 101 102 103 104 106 112 118 123 122 122 117 119 119 120 127 130 127 117 115 119 123 118 116 117 117 123 128 130 128 127 123 119 112 104 95 86 76 74 76 78 84 88 89 90 94 96 99 97 99 100 98 97 95 94 96 103 99 92 88 82 76 69 60 50 41 43 50 45 46 29 22 21 23 31 42 46 39 46 57 60 56 55 51 60 81 86 91 91 93 87 87 85 88 92 94 95 95 100 103 112 112 112 111 116 116 119 123 123 124 122 127 129 133 131 131 132 135 127 124 134 131 133 132 135 133 132 125 116 109 105 102 92 88 86 88 87 91 90 94 94 98 104 104 105 104 108 105 108 107 108 110 115 113 101 95 91 78 63 61 54 47 50 51 44 43 26 20 26 29 33 40 40 40 49 60 61 61 56 55 71 89 91 92 97 95 89 85 87 86 92 95 96 97 96 102 112 119 115 123 129 129 129 135 135 135 136 137 141 139 138 134 137 134 131 127 129 129 128 128 132 135 128 119 108 107 105 103 97 96 100 99 97 91 94 93 94 95 103 109 110 112 111 113 113 115 114 114 118 119 110 102 90 75 74 76 67 52 47 48 47 51 28 25 27 29 34 45 44 43 48 58 58 55 54 64 81 89 95 94 98 97 92 85 88 92 95 99 98 100 100 107 111 117 120 131 134 130 131 136 137 135 138 138 138 135 136 135 133 135 137 135 125 122 119 122 121 122 119 114 111 112 115 113 113 113 114 107 105 102 104 99 95 97 102 109 112 115 110 113 117 122 122 121 123 117 116 113 99 82 77 80 72 55 53 53 46 48 37 32 30 30 35 44 51 50 55 60 60 54 57 72 86 90 91 98 98 98 94 95 97 106 109 106 103 104 110 109 113 113 122 127 127 123 128 132 131 130 134 138 135 138 136 134 132 133 136 138 132 120 116 117 117 118 125 123 122 122 124 127 129 127 120 117 110 110 108 107 99 103 108 110 112 111 116 117 122 127 131 130 131 131 132 130 113 97 95 94 75 59 57 53 44 42 36 34 33 31 35 42 49 52 58 61 61 60 66 77 84 91 91 98 97 98 100 100 105 109 111 111 114 112 116 120 120 116 115 121 122 119 117 119 124 133 137 140 137 136 133 129 130 130 136 132 123 119 122 118 118 125 129 127 122 122 124 129 126 126 123 120 117 114 113 113 110 111 114 113 112 109 112 117 124 130 138 142 141 139 136 138 122 105 102 102 88 69 56 48 41 37 39 33 32 33 35 42 46 57 59 60 60 64 70 78 85 88 93 96 100 98 102 100 105 107 106 109 112 116 117 122 121 121 118 125 125 118 110 111 124 130 132 134 134 128 127 126 127 126 128 126 122 122 116 113 111 121 124 123 122 125 128 127 126 120 118 118 120 112 112 110 112 113 112 114 112 105 105 118 125 132 142 143 140 141 142 143 135 118 104 96 85 66 53 48 45 42 43 39 37 35 35 43 51 59 59 59 68 75 80 81 87 88 93 97 99 99 98 102 105 107 106 110 112 114 116 119 122 122 122 121 125 122 114 106 116 125 126 128 127 125 124 127 126 122 116 122 123 121 117 111 108 111 119 120 117 117 123 123 122 118 115 114 113 111 105 104 104 105 106 106 107 110 109 111 124 135 146 147 144 140 139 139 138 134 116 98 85 69 53 43 44 47 37 34 29 25 32 42 49 55 65 73 79 81 84 88 88 91 91 94 92 91 91 97 102 104 109 105 105 111 116 116 119 117 119 117 118 119 115 110 108 113 114 120 122 123 118 115 110 108 110 111 113 110 110 107 105 110 121 119 112 110 111 110 105 105 100 101 100 97 91 90 91 90 91 93 97 103 113 110 119 136 145 150 149 144 140 140 139 140 128 106 88 72 58 42 38 41 32 27 19 18 29 44 50 61 73 78 79 86 87 89 90 90 91 87 81 75 77 81 84 84 89 92 94 99 100 106 110 112 109 111 112 117 115 111 105 99 99 106 118 118 113 106 107 107 104 100 99 102 104 109 107 114 118 114 102 100 98 91 87 83 80 79 79 73 72 69 69 69 74 80 83 88 98 106 120 138 145 147 148 144 138 139 139 139 130 107 86 70 58 39 31 43 30 22 19 24 32 41 48 65 80 83 83 86 92 93 90 84 78 66 57 55 64 66 64 68 69 71 73 78 82 90 90 94 98 106 108 109 112 108 105 95 98 109 114 111 105 97 94 97 102 105 100 100 94 96 100 109 112 106 94 86 83 76 71 67 63 60 60 52 50 51 52 57 61 71 76 73 72 82 102 126 139 147 152 147 141 139 141 142 137 112 84 67 55 33 20 46 27 23 24 28 31 34 48 69 81 83 85 88 96 97 88 71 56 50 49 54 58 58 53 56 52 47 47 48 54 60 69 72 81 88 97 100 101 102 94 88 92 105 109 105 96 93 87 87 94 93 90 88 89 83 86 93 96 87 80 71 62 59 54 49 46 48 49 50 49 50 53 64 71 75 77 74 76 76 91 115 134 142 145 143 139 136 134 139 138 113 82 64 53 32 13 49 28 26 24 21 24 33 50 70 82 85 87 93 100 92 73 62 59 63 68 68 66 63 57 50 50 45 41 36 37 39 46 47 52 66 78 86 86 87 84 85 87 98 102 96 89 89 91 91 92 90 88 84 84 80 79 80 78 73 70 66 58 51 43 43 43 46 49 51 57 63 72 83 94 99 99 99 103 101 101 108 119 131 138 139 138 137 137 142 137 112 81 61 49 36 34 65 31 32 25 18 18 28 47 66 79 88 89 95 96 81 66 69 78 88 98 99 102 98 94 90 85 76 65 53 42 38 36 36 38 47 56 66 73 75 78 79 85 89 95 91 91 91 94 95 96 95 86 83 79 83 80 75 69 68 66 61 60 60 60 59 59 59 66 72 83 90 98 99 93 92 90 96 101 109 109 107 110 120 132 139 142 141 137 139 137 111 80 63 49 39 48 64 37 38 42 32 17 23 44 64 79 88 89 90 89 82 80 82 88 90 90 92 93 95 94 96 94 87 76 70 64 58 57 57 55 52 52 58 64 63 70 73 79 82 82 90 96 100 101 104 102 98 88 80 76 75 73 68 64 62 65 67 73 73 71 64 60 62 62 65 67 69 76 82 84 85 84 87 94 97 95 93 100 108 123 137 143 143 140 141 138 111 82 69 57 48 50 60 37 31 36 24 10 21 37 59 83 89 89 89 85 77 72 75 82 82 79 84 84 85 84 81 77 70 64 60 60 58 58 61 64 62 62 62 60 62 62 70 74 84 84 91 99 108 105 108 111 101 91 82 81 72 66 63 68 73 82 84 74 64 65 64 58 61 64 59 63 64 58 52 56 66 81 86 92 92 92 93 100 110 118 131 140 142 143 143 138 117 89 72 60 58 55 51 41 34 31 16 6 14 30 57 82 92 91 91 84 75 71 72 75 74 78 81 77 69 61 55 56 66 68 57 50 47 52 58 60 59 58 64 62 66 70 70 74 78 85 89 100 108 111 106 105 102 96 91 81 73 68 70 74 82 80 72 61 65 71 65 48 35 32 31 33 34 36 36 46 52 54 60 71 84 90 94 98 102 112 122 134 138 138 139 139 120 91 75 66 64 65 51 48 41 32 18 10 17 32 58 82 93 94 91 84 80 79 77 78 76 77 70 52 42 38 37 37 35 29 25 24 29 37 52 65 66 62 61 60 56 65 75 79 77 83 90 103 114 116 110 102 103 101 98 85 77 74 72 68 71 65 57 70 75 68 55 36 29 31 29 23 17 23 33 30 40 52 46 46 65 81 89 92 99 107 114 127 138 136 136 137 125 98 81 78 77 65 50 56 46 34 20 18 24 36 59 84 94 91 87 87 81 78 77 76 71 56 39 30 23 22 31 24 16 23 27 27 26 37 57 68 69 75 75 73 63 58 70 77 82 82 93 101 109 109 109 103 97 103 102 90 79 77 70 72 77 81 84 76 71 100 94 37 45 83 64 38 22 44 86 72 32 29 38 42 49 64 78 88 94 102 113 123 129 129 132 137 127 104 87 76 74 58 45 63 54 41 28 23 30 44 64 83 95 91 87 87 86 81 81 78 62 38 25 14 17 66 94 47 19 62 69 41 27 38 82 96 75 74 86 94 94 75 66 73 80 83 89 96 99 98 94 96 98 104 101 88 80 76 67 83 108 105 86 68 89 138 112 43 38 46 33 28 17 46 106 121 80 42 33 40 49 61 80 87 95 101 111 119 122 124 131 137 128 108 83 69 70 54 41 65 65 53 39 24 35 50 68 88 97 92 87 84 84 84 78 69 48 28 17 21 60 109 100 42 21 51 53 38 31 35 90 120 93 74 84 103 108 91 72 73 77 80 82 91 98 103 97 95 98 96 93 83 83 72 70 94 113 94 77 80 98 130 135 80 33 27 26 18 24 75 122 119 87 74 79 71 73 79 86 91 96 100 102 109 117 126 133 140 131 107 78 67 72 64 50 61 71 60 50 35 41 55 68 90 97 91 89 87 84 84 82 76 60 41 41 52 75 98 105 70 28 12 15 23 30 63 118 124 95 90 89 85 100 111 89 75 78 83 88 89 94 95 93 92 98 102 99 86 79 72 76 104 103 84 78 77 75 89 118 119 76 45 34 38 75 109 110 89 79 103 126 115 99 91 91 96 98 101 102 107 123 134 139 141 134 107 78 68 77 67 48 61 73 65 52 40 37 51 69 92 95 90 89 90 89 86 86 80 71 60 60 64 73 93 114 116 82 45 31 28 61 108 114 92 76 83 95 98 104 108 94 78 80 79 88 93 97 95 100 110 108 104 98 90 79 70 75 98 108 113 110 93 84 85 93 100 96 82 72 81 93 90 78 76 94 118 124 121 116 108 99 103 108 113 114 117 129 138 142 141 130 99 74 70 83 71 47 60 71 70 54 34 34 51 74 91 96 90 85 87 86 85 84 79 76 79 76 73 74 83 93 105 108 100 88 82 88 91 88 85 88 94 109 114 105 98 86 79 78 80 87 97 103 108 109 115 114 105 99 91 83 75 78 85 101 114 117 109 100 99 99 91 83 81 73 72 69 70 76 86 103 116 123 127 125 115 111 108 110 112 109 111 125 138 139 138 125 92 74 83 89 67 44 69 77 71 57 35 32 49 70 93 98 89 86 92 93 93 91 90 90 94 89 82 74 67 68 76 87 87 85 85 86 87 93 93 95 103 104 105 101 94 87 84 76 77 87 95 104 114 115 109 104 100 101 91 82 82 82 85 92 100 102 103 105 109 107 102 91 85 83 83 81 84 92 99 111 117 125 130 124 114 113 117 117 116 109 110 120 130 131 139 124 89 84 93 92 61 42 68 80 74 60 37 26 45 64 87 94 93 88 90 92 94 98 99 99 96 94 89 84 78 80 81 82 79 77 84 93 98 95 96 100 102 103 106 100 92 90 85 74 73 90 98 104 108 110 105 96 100 101 94 80 80 83 88 89 94 96 93 93 101 109 114 113 106 107 105 105 104 105 110 116 120 123 124 119 115 113 118 116 114 111 116 129 133 134 133 115 84 82 91 88 53 48 82 75 75 67 42 21 46 69 86 94 90 89 88 93 100 107 102 97 96 98 96 94 92 90 95 93 95 96 100 105 108 103 102 106 105 108 105 101 95 94 84 69 72 86 98 101 102 105 107 103 103 102 96 82 75 83 88 93 93 99 101 95 92 100 103 108 114 115 116 111 110 109 109 112 117 121 118 112 111 113 115 115 109 110 121 131 131 131 128 111 87 79 89 78 49 57 103 81 79 74 45 19 45 84 90 92 89 88 87 94 102 103 102 100 102 100 100 98 97 98 101 103 103 106 107 111 110 110 106 106 109 108 102 97 99 96 85 68 75 85 91 88 91 95 95 98 91 97 105 97 76 80 87 95 98 97 100 105 104 98 97 98 104 107 108 105 106 107 106 111 112 116 113 112 110 117 120 118 112 111 123 131 134 132 130 110 83 71 90 74 47 65 114 98 86 75 45 14 38 83 96 91 90 85 86 93 102 104 103 106 103 99 96 98 98 100 99 99 103 109 109 109 107 108 111 109 110 102 101 99 98 94 85 74 73 83 86 88 88 92 93 96 95 99 110 105 86 78 88 96 102 102 101 105 110 114 111 105 103 102 100 102 103 103 102 105 105 106 108 107 112 113 116 115 116 119 127 135 137 137 129 97 70 79 89 62 47 68 117 118 92 78 47 11 31 73 93 93 91 89 89 94 97 103 107 109 106 104 97 100 103 101 101 100 102 102 103 104 109 111 113 115 109 103 103 105 99 91 82 80 78 87 93 94 94 98 102 98 99 97 104 104 95 85 89 95 101 105 105 105 105 115 116 115 111 108 105 103 102 103 107 109 112 110 115 113 116 116 114 110 114 122 128 136 135 140 124 78 65 96 82 45 52 72 95 119 99 82 48 10 30 70 86 89 90 94 91 93 96 98 104 107 112 110 107 101 99 97 98 99 98 100 105 110 111 112 111 108 104 103 104 104 97 87 85 83 83 87 94 95 102 102 105 106 101 99 107 116 103 92 88 91 93 102 109 107 106 108 114 117 116 114 116 112 112 112 113 115 115 117 115 117 117 116 110 110 114 119 126 129 132 142 116 68 85 106 63 40 61 79 83 108 104 86 52 15 27 63 82 88 89 92 93 92 93 93 104 108 111 111 111 108 105 104 100 100 99 103 108 112 115 111 105 105 106 102 104 98 90 85 84 80 83 87 90 94 98 102 105 110 101 97 109 118 108 97 91 87 87 97 106 107 112 110 111 110 111 113 118 114 113 117 116 118 116 118 116 115 110 106 110 114 118 121 120 120 133 139 112 89 105 101 50 43 65 82 91 93 99 88 48 15 25 47 74 88 91 92 90 89 88 94 103 112 115 119 114 114 111 107 104 103 104 107 113 115 116 109 107 104 107 104 99 92 87 88 84 84 85 89 89 95 102 107 103 97 96 92 98 103 106 101 97 91 93 96 99 109 112 115 113 112 107 113 113 113 115 119 119 118 119 116 117 116 115 107 111 115 113 115 117 121 131 136 113 102 112 88 45 53 76 90 106 93 98 84 46 18 25 38 61 89 93 90 91 92 89 91 97 107 114 115 116 115 114 116 119 113 116 122 120 114 109 106 107 109 108 104 95 89 90 90 86 83 84 89 90 91 98 95 95 93 92 89 91 100 106 106 103 102 100 101 103 109 110 114 115 115 115 111 109 108 113 117 120 119 118 116 113 108 106 107 108 111 111 112 112 118 136 130 106 137 144 80 37 60 87 90 110 91 94 85 42 15 28 38 66 94 92 89 90 89 93 91 94 99 111 111 117 119 120 119 115 119 121 124 114 109 108 111 109 111 109 101 96 91 91 86 85 81 85 90 96 93 90 93 96 100 92 91 92 103 108 112 109 101 98 100 107 108 113 113 117 115 116 112 113 111 113 117 117 118 116 120 110 105 102 107 108 108 109 110 113 119 135 105 101 163 139 66 44 73 89 90 115 90 88 76 36 21 36 42 62 92 98 92 90 86 90 90 90 95 99 104 109 115 115 116 113 118 120 116 110 110 112 111 114 112 109 100 98 94 90 86 82 79 84 92 91 94 88 94 99 101 96 95 100 106 113 114 108 97 93 91 97 108 115 117 114 117 119 118 116 118 116 118 116 116 117 119 118 111 109 104 106 106 106 106 109 119 129 103 121 182 181 95 48 77 84 111 116 94 87 73 38 19 36 34 53 115 117 88 86 87 86 88 87 92 92 99 104 108 114 116 116 113 115 110 112 111 113 112 112 112 111 108 99 92 86 85 79 85 89 94 95 94 93 95 101 100 103 100 101 102 113 124 111 91 83 82 88 108 112 118 118 120 120 120 117 118 116 114 117 115 118 117 118 111 107 104 105 107 101 105 107 119 124 105 144 185 136 60 54 69 98 144 115 95 82 60 29 20 36 21 74 186 151 85 83 87 85 84 89 90 94 93 103 113 115 116 119 115 111 113 110 113 113 115 111 114 115 114 105 92 82 77 78 82 92 94 96 95 100 100 102 103 106 105 101 105 116 133 117 93 78 69 77 96 110 116 122 121 119 116 117 117 115 114 116 118 118 118 113 112 108 110 102 101 102 103 107 120 124 125 147 121 66 49 55 74 120 140 115 91 75 51 23 23 40 20 15 125 180 108 78 84 86 84 88 87 90 95 104 113 113 116 115 117 112 115 114 117 117 116 117 116 117 111 101 88 82 75 76 84 92 94 96 97 94 96 104 109 107 106 104 109 123 142 133 101 82 68 68 83 101 108 113 114 116 118 118 116 117 118 112 114 112 114 110 111 111 107 104 99 101 101 106 117 127 169 184 95 40 45 71 103 116 108 99 82 65 46 27 24 32 22 4 39 115 107 82 84 81 83 83 90 90 95 102 114 115 116 115 115 115 112 115 111 115 113 117 112 104 100 92 83 79 80 81 89 90 95 95 99 101 103 105 104 109 108 109 111 124 145 142 115 95 75 64 74 89 103 105 113 113 116 116 114 113 112 112 111 117 115 111 106 108 106 106 104 103 104 106 130 133 89 56 54 57 76 99 98 96 102 82 73 60 44 38 34 34 31 21 11 24 64 83 85 85 87 85 89 94 94 102 109 110 109 111 111 111 110 110 110 111 114 112 108 100 99 88 79 78 82 85 89 91 94 100 105 111 113 116 115 118 117 113 115 126 139 137 128 115 85 66 77 84 94 103 110 116 114 117 117 117 111 111 112 115 108 104 105 104 105 104 102 102 101 112 147 128 43 17 48 72 87 89 89 119 140 79 73 68 64 57 52 46 40 33 22 11 40 79 85 83 84 89 86 92 93 97 103 110 111 106 109 110 113 110 113 113 113 106 102 96 95 91 83 82 81 85 90 94 94 101 109 118 131 135 133 131 129 117 115 125 132 125 122 120 96 76 81 92 92 102 109 115 112 114 117 114 112 109 109 102 101 96 102 103 103 105 101 101 100 118 156 126 43 27 45 70 90 95 104 142 154 82 76 68 59 50 48 39 30 29 21 9 28 72 83 81 85 86 90 95 99 98 100 106 110 106 109 112 114 114 110 111 109 106 96 96 101 104 91 83 82 81 85 90 92 97 109 121 126 124 121 120 117 110 107 115 117 111 105 100 87 83 88 94 97 99 107 112 115 111 111 109 110 108 104 102 99 101 101 104 103 103 101 102 99 116 162 124 35 21 46 68 96 113 113 149 171 78 65 52 38 35 38 33 28 26 19 6 21 64 85 85 87 85 89 94 101 98 96 100 105 110 110 113 112 113 108 111 108 101 98 100 107 107 99 89 87 84 86 87 90 96 102 105 103 104 101 100 101 105 105 105 104 101 96 89 86 90 96 95 100 99 103 109 112 111 109 107 105 108 105 109 102 98 97 101 105 104 103 103 99 126 167 112 24 12 38 65 94 116 139 174 172 71 52 41 36 37 36 33 35 33 23 8 12 54 82 83 84 86 86 94 97 95 90 96 101 105 109 110 110 108 110 109 108 101 103 107 108 106 106 98 87 82 76 75 79 88 94 99 99 98 94 93 99 100 100 95 91 80 81 88 96 99 98 98 98 102 102 106 105 109 109 108 109 108 111 111 105 97 96 96 103 100 99 104 103 130 165 100 10 3 20 52 93 125 138 137 129 62 46 35 32 32 37 42 42 30 17 11 9 41 78 85 84 84 86 90 91 90 89 94 103 102 103 108 112 108 109 109 106 104 107 108 105 105 100 100 97 89 77 67 69 77 87 93 95 92 91 87 93 94 90 80 74 80 87 97 101 101 98 103 104 101 101 107 110 111 109 108 111 108 110 108 103 100 101 103 103 101 102 108 110 137 153 84 10 3 10 25 52 72 66 82 129 52 40 34 32 39 45 43 39 28 19 16 8 29 72 85 84 83 85 83 86 87 91 94 99 101 101 106 99 99 103 111 108 108 107 107 104 102 99 96 97 96 95 80 73 74 80 85 84 82 83 90 92 86 76 77 81 89 93 99 102 102 102 102 105 100 103 104 107 106 104 105 106 107 106 107 100 101 101 103 101 101 105 106 111 143 146 69 33 27 11 8 18 29 46 90 133 49 37 31 38 48 47 41 37 26 22 21 10 16 63 85 84 85 85 86 86 87 88 91 98 100 101 106 94 93 104 109 110 107 107 105 105 100 101 97 95 96 95 87 81 84 86 86 83 79 83 90 87 83 85 89 93 91 94 95 100 103 105 104 104 102 101 103 104 104 104 106 105 107 105 103 101 102 104 103 103 102 106 103 112 147 129 52 44 61 45 26 26 45 73 94 101 39 31 37 48 46 42 40 36 27 25 20 9 13 56 87 85 85 85 87 83 85 85 91 96 95 98 103 107 104 108 106 109 108 107 107 108 105 96 94 90 96 94 90 87 91 96 99 97 87 87 93 98 96 97 94 99 96 96 100 102 106 106 110 104 103 100 104 102 104 105 105 105 102 102 98 100 101 105 103 104 104 105 106 116 146 106 31 39 65 86 68 26 25 52 105 137 32 38 45 46 44 42 41 36 32 28 20 10 12 44 79 84 83 86 84 86 82 84 91 96 95 97 101 103 104 103 108 109 110 106 108 109 109 100 96 96 94 92 91 93 93 100 104 102 92 91 101 102 100 98 97 98 96 95 99 99 100 102 106 103 97 102 103 104 101 104 101 96 92 94 96 94 100 105 107 106 105 104 103 123 144 79 19 46 65 95 113 70 36 36 59 92 41 45 50 48 46 45 43 35 30 24 18 13 10 35 70 85 83 84 83 84 84 87 91 92 93 93 98 97 102 102 106 105 106 104 104 105 105 105 97 97 93 92 91 96 100 102 103 99 95 95 103 100 104 105 102 96 96 98 98 99 99 104 102 102 98 100 100 102 101 101 103 95 90 92 97 92 99 105 105 105 107 105 107 130 136 64 14 47 68 88 130 138 90 50 48 53 50 53 50 45 45 49 43 33 27 23 20 20 18 32 69 85 83 86 89 82 81 83 88 86 87 92 94 99 101 105 104 105 100 104 106 105 102 96 93 90 89 89 95 99 103 100 100 100 94 97 101 103 98 100 98 97 94 97 100 101 101 102 103 98 99 98 101 98 99 96 97 96 91 91 92 95 97 104 105 104 103 103 114 137 122 45 18 51 76 89 124 158 163 123 71 51 48 48 46 46 48 49 38 31 30 26 24 23 27 34 62 87 87 84 86 85 81 82 85 88 87 87 88 95 99 104 103 102 102 99 100 101 102 96 94 93 93 88 90 91 95 94 96 101 93 96 101 99 92 94 92 88 88 89 96 97 97 97 102 95 94 96 96 93 91 93 91 93 91 90 89 93 97 102 106 108 107 105 118 143 105 30 23 57 82 92 123 154 172 174 147 103 57 55 55 54 52 41 37 40 39 31 25 23 30 29 40 75 88 86 84 85 81 84 85 85 81 79 84 87 96 100 103 98 96 94 94 96 95 91 94 98 90 84 83 87 84 88 93 95 91 90 97 97 97 91 89 84 85 90 93 89 89 92 91 93 88 92 91 94 92 89 88 89 91 86 89 88 92 99 104 105 105 106 123 139 88 22 22 61 88 95 120 163 182 163 156 171 63 62 57 53 49 41 46 47 35 31 29 31 38 28 24 59 83 84 82 82 85 83 83 82 81 79 80 83 90 97 101 96 93 92 90 92 89 89 91 93 90 89 93 96 98 97 99 92 90 92 99 106 103 96 93 96 91 96 99 91 85 90 92 91 85 88 92 92 94 86 86 87 87 87 89 88 92 98 102 105 107 108 127 129 66 22 30 59 95 108 125 154 176 178 168 183 72 65 57 55 50 52 47 36 32 32 29 31 38 28 23 48 83 88 86 82 81 81 80 79 78 81 79 85 88 98 98 99 92 91 90 88 86 83 90 92 95 92 96 95 96 100 97 94 94 102 107 108 93 87 85 86 83 84 92 96 96 91 91 87 87 85 87 84 87 85 81 84 86 90 87 90 89 96 104 107 107 111 133 113 43 24 38 63 104 120 125 151 164 169 177 174 86 72 60 61 58 48 41 37 32 31 34 42 43 32 18 30 72 88 85 86 82 81 81 82 79 79 81 81 88 94 94 92 89 87 87 85 83 83 85 84 79 78 78 79 76 77 72 80 93 101 97 83 75 70 71 69 69 67 73 76 78 82 84 88 83 82 83 84 82 82 79 82 87 90 87 87 93 96 102 108 110 120 136 88 21 20 37 62 106 147 140 145 183 183 178 176 92 75 71 77 64 48 41 39 36 32 40 52 46 36 18 13 52 87 85 84 81 80 81 80 80 78 82 77 83 87 87 85 87 87 84 86 80 75 69 73 69 69 69 70 69 65 62 61 71 68 67 65 68 66 66 69 63 60 60 61 58 62 61 65 67 76 83 85 87 85 80 81 92 94 93 89 92 100 108 110 113 131 122 55 16 26 43 65 97 146 168 152 173 209 206 182 91 88 84 67 49 42 38 38 36 38 48 52 44 34 18 9 38 76 86 82 84 80 77 79 79 77 79 80 78 83 84 86 84 81 74 65 59 55 59 60 61 65 80 83 86 85 79 69 65 66 68 77 85 90 90 88 61 44 58 67 58 46 35 31 50 62 76 81 84 78 75 82 91 99 100 97 92 100 106 108 117 140 99 23 18 32 43 59 85 127 176 165 113 144 189 183 95 100 81 51 43 42 40 39 36 43 50 46 41 29 22 18 19 55 83 85 85 80 77 80 79 77 78 83 83 86 85 84 76 63 46 27 19 16 43 71 68 63 73 80 86 99 105 103 88 85 93 98 88 78 66 59 71 78 82 89 70 34 29 47 72 81 79 78 77 79 80 91 96 100 100 99 97 101 105 105 125 129 58 9 23 29 33 48 82 123 151 161 134 108 122 132 71 73 67 58 55 47 41 37 43 51 51 49 37 27 28 19 6 33 74 89 84 80 78 80 79 79 86 84 87 83 80 76 73 71 66 46 23 20 54 88 88 88 77 54 48 48 43 41 42 40 36 37 52 65 58 65 112 120 90 73 53 37 48 73 86 88 79 75 75 80 86 92 99 100 102 97 99 101 104 112 126 96 29 13 23 30 33 39 78 137 145 140 145 134 116 102 85 89 91 76 58 46 39 48 57 55 51 44 30 27 26 18 7 11 54 86 86 84 81 81 83 84 88 89 88 84 78 76 77 90 90 72 51 41 41 49 75 108 111 101 97 82 62 47 40 44 54 65 95 122 88 72 117 116 67 47 48 56 67 82 86 81 80 73 71 75 83 89 93 97 101 102 104 104 105 120 119 58 19 18 20 32 42 39 64 128 150 138 134 134 124 119 106 101 94 75 56 44 48 59 57 52 46 32 27 25 21 20 16 2 26 77 90 85 82 80 84 86 84 87 84 86 82 76 77 84 85 77 67 58 51 39 58 88 89 89 110 105 96 107 87 86 111 95 96 125 87 54 74 69 54 57 64 63 77 82 83 78 77 74 76 79 83 89 90 99 104 106 105 103 113 126 88 29 17 18 23 35 43 44 59 108 154 149 123 115 112 101 102 101 87 65 50 53 64 60 49 46 35 25 28 30 28 23 12 2 11 54 86 89 82 82 84 87 89 87 86 84 82 80 82 78 80 78 69 65 65 61 53 58 55 59 75 76 71 87 76 75 96 80 70 78 67 48 48 53 61 68 69 68 77 84 82 82 78 76 75 79 86 86 92 101 105 101 103 105 127 115 41 14 24 21 23 32 40 46 54 82 118 141 136 107 95 86 102 90 65 57 66 70 60 46 43 33 24 28 32 29 22 13 5 4 7 34 73 86 84 81 87 90 90 85 88 83 80 78 78 78 76 77 70 66 65 73 69 65 63 59 58 57 57 54 53 54 59 56 56 55 56 61 67 74 71 70 70 78 82 85 78 81 82 78 75 79 84 84 93 103 104 99 100 111 126 87 19 8 23 25 24 33 39 44 53 71 91 106 114 108 93 79 78 65 72 82 70 49 37 40 37 29 31 34 24 16 14 9 7 10 5 11 51 85 89 86 83 84 84 85 80 81 80 80 78 79 75 71 71 68 67 68 73 73 75 73 77 72 67 65 64 64 65 70 70 79 76 79 79 83 77 69 73 81 88 81 80 77 76 71 72 74 81 90 94 101 99 98 100 116 115 73 31 18 25 28 25 34 41 37 45 59 74 91 94 93 84 69 78 86 84 76 61 51 46 43 38 34 34 28 20 16 14 13 13 15 11 5 31 74 89 87 84 82 81 84 82 80 77 76 78 80 80 74 71 67 70 66 73 82 80 87 88 88 82 82 78 81 81 86 90 91 89 84 81 73 70 70 80 84 85 83 83 82 80 77 75 76 78 86 93 101 100 98 107 117 94 79 79 53 28 23 25 34 41 38 39 49 57 65 76 73 68 74 84 86 82 76 67 57 60 61 56 50 42 30 23 19 18 18 17 18 16 17 24 53 82 86 84 82 84 82 84 79 78 76 79 80 78 76 67 67 69 74 74 80 80 86 90 93 89 86 85 83 90 92 95 89 88 79 72 70 72 82 86 85 78 79 79 80 77 80 78 80 86 84 92 98 99 99 114 110 69 51 74 74 45 28 26 28 35 37 34 45 51 52 59 59 66 73 83 80 76 66 59 69 83 83 76 65 50 39 32 31 34 42 38 24 18 18 13 31 69 86 85 83 83 82 81 81 77 75 74 75 76 75 69 68 70 76 76 77 82 81 86 83 85 83 86 82 82 85 83 82 77 74 70 78 80 83 83 81 77 77 79 76 78 79 78 81 86 87 93 97 97 104 114 102 63 39 54 65 57 45 36 33 35 38 33 38 54 61 61 60 60 56 77 63 62 67 80 88 91 88 70 66 64 48 39 39 49 50 38 25 21 22 25 35 62 84 84 80 81 84 80 81 77 78 73 74 77 75 75 69 69 66 76 79 84 85 86 84 81 83 79 79 79 81 77 82 79 81 80 76 77 78 82 77 78 73 76 78 81 79 77 81 85 90 95 98 98 108 103 91 77 61 55 64 69 55 46 45 40 36 34 27 40 58 56 52 53 51 91 56 43 63 74 68 67 64 54 54 53 44 39 38 32 28 30 31 31 35 35 42 58 76 84 82 81 81 85 83 79 79 80 77 76 74 75 74 71 74 71 72 76 80 76 81 79 79 81 84 84 77 76 76 79 76 77 75 77 77 79 79 77 76 77 84 79 79 80 83 89 93 96 97 104 104 95 86 81 79 75 78 85 78 69 77 86 72 54 41 29 36 55 59 54 49 111 78 51 41 36 44 55 55 52 50 51 50 43 33 26 27 30 35 35 38 40 49 61 70 83 85 83 80 85 82 81 78 79 77 72 71 69 77 77 80 75 77 73 73 72 72 74 73 80 76 76 74 78 76 76 77 79 79 76 74 72 78 75 76 78 83 79 77 82 86 94 97 99 100 104 97 90 87 85 88 79 72 89 95 93 101 113 105 73 72 68 48 55 67 65 64 93 99 89 68 57 55 58 62 59 60 59 52 40 31 29 31 35 37 42 43 47 53 64 73 79 84 82 83 84 83 82 81 74 74 73 72 71 69 74 76 80 79 81 80 82 79 77 77 78 77 75 78 78 79 75 78 78 80 75 68 73 71 73 74 80 79 80 82 85 87 94 99 99 102 93 89 86 87 87 96 95 68 57 78 99 120 138 137 104 81 103 108 90 87 83 96 112 108 88 67 58 66 68 63 62 60 55 49 36 24 30 32 35 37 41 47 49 57 67 76 78 81 82 81 76 80 82 80 77 73 74 72 74 70 72 75 78 81 86 87 86 83 79 79 78 78 76 78 76 78 78 74 75 75 75 72 74 70 73 77 81 84 84 84 85 91 93 99 100 94 85 87 89 89 91 99 108 85 35 42 97 126 149 181 183 148 129 129 112 94 97 115 109 96 81 69 66 68 66 68 72 70 69 61 40 26 27 31 35 42 44 49 49 58 67 74 80 81 85 78 80 79 75 74 77 74 71 73 72 72 70 75 73 79 85 89 86 83 82 85 90 85 85 81 83 82 83 77 73 72 72 76 73 73 74 78 73 77 79 86 89 91 97 97 91 83 85 87 92 92 93 100 105 91 53 58 96 114 136 169 192 199 219 197 148 137 146 153 95 98 112 100 70 59 78 96 99 96 92 77 51 29 20 25 38 47 54 59 63 66 71 75 80 81 85 83 80 80 78 78 74 73 73 77 76 74 73 73 75 75 81 86 91 91 89 88 91 90 89 88 89 89 86 82 75 73 66 65 70 75 79 80 82 81 81 83 89 94 95 86 83 84 85 89 93 95 96 97 100 81 60 94 114 108 115 133 148 150 175 211 209 171 148 145 88 99 128 115 94 99 112 116 119 117 117 114 72 24 19 29 42 52 66 69 70 69 72 77 80 84 84 89 84 82 78 76 72 72 76 75 76 76 77 73 81 82 83 85 90 92 90 89 87 91 90 92 89 91 88 85 78 73 71 68 74 71 78 79 80 82 85 91 96 97 85 81 81 85 87 93 92 96 97 97 90 44 44 123 145 122 106 110 148 163 150 182 224 221 180 149 91 98 121 128 134 138 132 132 136 135 136 148 121 43 12 26 39 54 65 71 71 72 73 76 81 84 85 86 90 87 80 71 73 73 71 71 73 78 77 78 79 84 85 89 87 90 91 94 92 89 89 86 90 88 85 83 83 81 81 76 73 77 77 80 80 83 88 95 92 85 80 80 82 88 94 92 98 99 98 89 59 23 68 135 149 144 128 120 150 174 169 176 216 247 221 178 100 110 126 137 143 142 137 140 149 155 155 172 175 92 20 24 37 53 65 73 74 74 76 76 81 84 89 87 90 91 87 78 71 69 71 73 68 75 75 79 81 85 87 87 88 90 94 95 99 90 88 89 90 85 85 86 86 86 83 83 77 80 77 77 80 88 91 88 81 79 83 83 87 90 95 98 102 104 98 77 37 41 120 156 154 157 156 152 162 169 168 159 191 228 211 168
## 7052                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          177 176 174 170 169 169 168 166 166 166 161 140 69 5 1 2 1 18 61 96 110 122 129 129 127 125 125 119 112 110 111 107 102 102 99 96 98 95 91 90 91 94 92 88 86 86 86 85 86 87 84 84 87 88 85 85 86 84 87 97 96 96 99 101 105 107 111 109 94 68 31 3 2 48 141 164 137 123 124 120 106 100 103 107 114 115 112 113 112 110 114 113 114 114 111 111 176 174 171 169 168 166 166 164 162 163 158 107 24 0 2 0 7 50 98 119 128 134 131 132 131 129 125 120 116 110 113 111 104 105 106 101 97 92 86 88 90 92 92 93 91 86 88 87 89 89 85 88 88 87 89 88 87 88 89 93 97 101 101 100 103 109 110 109 106 93 63 20 0 7 83 155 141 124 125 119 105 99 102 106 110 110 110 110 110 111 113 113 113 113 111 112 173 171 171 169 165 164 163 162 161 158 136 57 2 2 1 1 25 81 116 126 133 134 133 132 129 129 125 123 122 116 110 109 107 104 103 102 98 91 86 88 90 89 87 88 88 89 88 87 87 88 91 90 87 88 90 92 91 89 93 97 99 100 100 100 100 104 110 112 115 114 92 49 9 0 27 107 132 123 122 121 108 99 104 106 108 109 113 113 109 112 115 111 111 113 111 111 170 169 167 165 162 162 161 159 160 149 84 11 0 2 0 6 51 107 130 134 138 138 131 129 128 129 126 119 116 116 114 108 105 102 100 101 99 95 93 93 93 91 89 90 88 90 91 89 89 91 91 88 86 87 87 87 88 88 89 90 91 94 97 97 100 103 106 110 120 122 113 87 37 3 4 57 115 122 122 121 107 101 102 105 111 111 114 114 112 111 112 111 110 111 114 111 169 166 164 163 161 161 159 154 152 122 43 0 1 1 4 28 86 134 142 142 142 136 129 130 129 125 122 119 111 109 111 106 106 103 102 100 100 99 98 99 95 95 95 95 94 94 93 92 91 91 86 86 84 81 84 84 89 91 92 94 94 93 97 103 103 106 108 109 115 121 120 104 63 16 0 20 81 117 121 117 108 102 102 102 108 113 112 111 110 107 111 111 111 112 110 108 168 165 163 160 159 156 156 151 138 80 12 0 3 5 14 52 107 136 143 147 144 136 133 131 133 127 121 121 116 112 108 107 109 106 103 101 100 102 98 97 97 95 97 97 94 90 91 90 89 87 86 86 85 83 85 88 90 90 92 90 94 95 97 100 102 104 107 108 108 111 120 112 87 43 4 3 42 95 115 118 106 97 103 108 108 110 112 112 110 111 110 110 110 112 111 112 164 162 161 160 157 155 153 148 118 51 7 2 6 13 24 68 111 135 144 148 148 145 140 138 134 126 122 121 117 112 108 109 107 109 106 103 103 98 98 96 93 94 92 94 95 93 91 91 90 90 89 88 84 85 87 88 91 93 89 92 93 96 98 99 104 102 104 109 111 115 119 121 97 57 16 0 26 83 112 119 108 98 103 109 110 111 115 115 112 111 111 112 111 111 109 109 166 163 159 159 154 152 152 141 91 28 6 6 8 18 44 88 120 138 148 151 146 142 141 139 132 127 122 123 120 114 112 111 113 112 105 100 94 93 94 94 99 92 89 89 91 93 92 91 91 93 93 92 90 94 93 94 95 96 95 94 97 96 98 100 101 103 105 106 110 113 119 130 111 71 33 5 12 67 112 123 113 100 100 104 110 113 113 114 111 110 111 113 111 112 111 112 163 162 158 158 154 150 150 124 61 11 4 9 13 31 65 101 124 140 152 153 146 142 140 141 136 131 125 125 123 116 112 111 113 113 107 102 101 97 95 97 99 95 93 93 95 94 95 95 94 95 97 96 93 94 95 98 98 99 98 101 101 98 97 97 100 100 104 109 110 116 125 136 127 82 39 12 2 38 96 118 110 100 101 105 110 113 113 113 111 112 115 112 110 111 111 110 161 161 156 153 153 152 147 106 40 11 9 12 21 45 78 102 119 140 154 152 148 140 137 140 140 137 129 125 127 123 118 113 108 109 108 102 99 96 94 95 99 101 94 93 96 95 94 95 96 95 94 94 91 88 94 94 96 98 97 97 96 96 100 100 103 101 104 112 112 116 126 138 138 102 54 23 5 20 69 106 106 97 99 104 109 112 112 113 110 111 112 113 112 110 113 111 161 160 158 155 153 150 137 84 24 12 15 19 29 53 79 100 123 145 153 157 155 142 134 135 136 131 129 129 124 121 120 116 114 110 109 102 96 93 94 96 97 97 95 94 92 93 93 93 93 92 92 87 86 89 94 99 98 95 95 95 96 97 99 104 104 107 112 112 112 117 133 143 142 115 69 36 12 9 45 91 103 96 100 105 108 113 111 110 111 111 113 112 110 109 111 113 160 159 158 155 155 154 133 62 20 20 21 29 37 60 86 102 123 149 163 171 171 153 136 132 133 130 132 133 128 122 119 117 117 117 106 100 95 92 92 94 94 96 95 93 91 91 92 90 91 88 87 87 87 90 93 95 94 93 93 96 97 97 98 100 103 108 111 113 114 120 134 149 147 125 77 43 25 15 32 78 101 99 102 104 108 113 114 112 111 113 111 111 109 110 113 110 159 159 156 153 152 152 129 56 21 22 24 35 44 74 93 98 126 154 170 176 168 152 135 133 137 133 132 132 130 126 115 110 113 112 104 99 95 94 89 88 93 97 98 94 91 93 93 93 92 91 91 91 95 95 94 92 91 90 91 92 94 96 99 103 104 104 106 108 113 120 134 149 157 143 96 55 37 22 30 70 102 100 100 103 108 113 112 112 112 111 112 112 113 112 112 110 158 158 159 155 153 152 114 45 23 18 21 41 62 80 86 95 125 154 174 178 177 169 149 140 136 129 129 132 131 126 117 111 109 106 104 101 99 94 91 89 92 95 93 90 88 93 95 99 96 94 98 94 90 90 90 91 89 88 88 92 90 93 97 99 102 104 107 109 110 120 136 147 155 148 109 62 42 31 31 63 97 99 100 104 104 110 114 112 112 113 110 110 111 111 112 112 159 159 157 152 150 146 110 46 23 28 35 50 64 72 78 95 126 157 174 178 181 181 163 147 138 131 124 124 134 131 120 116 106 104 103 101 95 87 91 91 93 95 93 90 89 97 99 97 97 98 93 93 93 91 89 89 87 85 85 87 90 96 99 98 99 98 103 105 104 120 140 155 162 153 118 73 48 41 34 54 89 98 99 106 105 109 113 109 109 113 112 112 112 117 114 109 158 159 157 153 151 143 105 45 23 32 41 50 68 76 82 105 137 160 175 185 185 181 172 152 140 130 124 129 132 131 125 116 113 110 105 100 93 92 93 94 92 88 88 86 85 91 93 89 91 90 88 87 90 90 88 85 85 81 83 87 94 100 102 98 95 98 103 108 113 120 137 155 165 157 123 84 57 40 38 48 79 98 103 105 107 112 112 109 109 111 113 112 113 112 111 110 157 158 156 153 153 144 98 39 28 41 46 55 71 79 88 109 142 163 179 186 183 182 183 173 154 132 120 124 131 131 128 122 116 109 104 99 95 95 90 86 87 89 87 86 87 88 92 92 88 90 92 93 90 86 87 84 82 84 89 91 94 96 96 94 97 100 106 113 118 123 133 156 169 167 138 102 72 47 34 40 67 91 98 101 104 111 113 111 111 109 110 112 112 112 111 111 160 158 154 153 151 136 80 35 30 37 51 65 74 83 95 119 149 172 182 186 189 184 180 175 161 144 127 121 124 124 123 120 117 106 101 102 99 94 89 86 88 89 86 86 88 87 90 90 92 93 90 90 88 82 85 87 85 87 91 92 92 97 100 97 96 99 105 112 116 125 136 157 171 171 155 126 90 56 38 42 66 88 96 101 104 112 113 111 110 109 111 111 110 112 112 110 158 156 154 153 150 129 76 36 32 42 56 66 80 95 111 132 149 161 169 185 188 180 176 171 163 152 143 130 121 119 121 122 118 114 107 105 105 101 92 93 92 89 90 91 90 90 90 90 90 90 89 87 88 89 92 91 86 88 94 95 97 98 99 97 99 105 109 112 122 131 140 156 169 172 165 139 102 68 44 42 66 87 95 101 107 111 110 111 113 112 112 112 113 111 111 112 158 157 155 153 150 138 90 39 32 39 54 69 85 95 114 138 152 165 181 188 187 189 181 169 166 158 145 135 126 120 121 124 118 115 110 108 104 100 103 105 99 95 92 90 91 93 94 95 93 90 90 91 93 96 98 99 98 99 99 99 102 105 104 103 102 103 105 108 114 124 132 149 164 169 170 155 120 78 52 42 59 84 96 104 105 109 113 113 111 113 111 111 113 111 108 110 157 159 156 154 153 140 85 36 32 40 51 69 86 96 116 142 156 165 174 182 187 192 188 173 162 158 152 142 136 135 128 123 125 122 118 115 111 110 109 107 99 99 101 95 94 99 95 96 93 90 93 93 96 99 99 100 98 98 102 106 107 108 108 105 103 104 107 107 113 116 120 136 157 168 169 167 146 105 66 46 50 76 93 97 100 108 110 113 113 113 113 111 110 112 112 109 156 156 156 152 152 132 75 33 38 49 58 75 96 111 129 146 161 169 176 186 190 190 189 176 159 152 147 144 138 131 128 128 127 125 120 115 115 116 113 108 102 101 102 99 101 97 95 97 94 93 93 94 98 100 101 98 99 98 99 105 109 109 110 112 113 111 115 117 115 117 122 126 139 158 171 176 170 140 88 54 57 83 96 97 103 110 111 112 110 110 112 111 110 112 109 109 153 155 156 154 152 133 76 37 41 56 70 86 106 121 133 148 159 174 186 189 188 189 186 171 152 139 133 135 139 132 129 130 126 128 125 118 113 118 118 111 104 102 104 105 104 102 102 101 100 100 98 100 100 101 105 109 107 105 106 110 111 116 120 120 121 123 122 119 111 110 117 124 131 145 166 174 173 161 112 64 55 78 94 99 103 106 108 107 109 112 113 111 109 109 111 111 155 156 155 151 149 126 72 42 46 62 74 91 109 126 143 158 177 186 190 191 190 185 174 157 139 133 135 135 135 135 129 128 126 127 129 121 113 114 117 114 110 105 108 109 110 108 108 107 105 105 103 100 106 109 112 113 109 113 115 114 116 115 117 124 122 116 116 113 108 101 103 113 120 129 147 166 172 171 140 79 54 72 91 100 105 107 109 110 110 108 107 110 108 108 110 108 156 155 153 149 146 129 82 49 52 63 79 96 115 135 152 168 181 188 190 190 187 175 154 144 138 139 142 135 131 132 127 126 125 125 125 119 116 114 114 114 111 107 106 108 112 114 113 111 107 106 106 108 108 110 114 114 112 116 118 115 114 112 108 110 110 105 102 96 94 96 97 103 107 116 129 155 169 171 157 96 55 66 88 100 102 106 110 110 106 109 108 108 110 109 109 108 152 152 152 148 146 133 85 50 53 63 79 94 108 133 162 179 184 185 187 188 176 153 147 147 146 144 143 139 133 130 130 132 127 125 124 124 119 115 114 113 109 109 112 111 113 118 119 115 113 112 113 110 110 111 112 117 115 113 113 109 103 101 99 93 87 79 75 68 72 83 88 99 105 115 123 135 153 167 167 114 68 74 93 101 102 105 105 106 106 108 109 107 108 108 109 107 154 152 151 147 145 137 92 54 56 67 79 93 116 153 177 182 185 185 185 181 164 152 154 160 153 142 140 136 131 129 128 127 127 126 122 119 115 111 109 107 105 110 109 108 112 117 117 115 110 112 113 108 109 107 108 114 110 102 100 97 89 85 72 62 52 50 57 61 67 73 72 76 86 107 124 124 131 159 170 125 69 68 88 98 101 104 105 107 107 105 107 106 106 107 106 104 151 152 151 147 148 136 90 56 60 69 80 96 122 159 180 183 183 183 179 167 162 162 160 160 151 135 128 123 123 115 113 112 110 117 118 109 106 101 100 105 105 105 103 105 109 107 111 109 108 110 108 106 105 102 97 99 96 88 85 79 73 65 50 42 41 37 47 63 69 70 71 77 82 88 99 123 132 146 169 136 68 59 84 99 101 102 102 106 105 104 105 106 106 103 104 101 150 152 150 146 141 130 90 58 63 76 84 94 122 164 181 182 180 179 170 162 163 167 165 154 143 130 118 110 104 96 88 80 79 86 86 83 84 86 88 92 98 100 101 97 95 95 100 103 104 106 109 106 99 96 94 90 85 79 75 71 61 49 45 48 47 48 55 63 65 72 80 84 90 96 100 119 129 133 156 138 75 59 85 84 80 97 112 108 103 102 107 106 99 102 103 102 146 148 148 143 140 128 91 60 68 79 79 93 130 167 181 180 177 170 165 168 176 181 165 142 129 112 101 91 77 68 58 50 56 61 60 61 63 71 76 77 83 85 89 94 95 97 99 93 92 98 100 99 97 92 88 83 79 75 68 58 48 46 50 50 54 59 62 68 72 71 74 78 78 85 99 118 129 137 156 145 81 59 67 60 82 124 143 125 106 102 102 104 103 103 102 104 145 145 144 143 142 136 102 60 64 77 80 93 130 171 181 178 173 165 164 179 190 176 148 121 106 97 85 74 58 46 39 36 38 41 44 46 52 58 64 70 74 78 84 94 95 96 96 94 93 94 91 87 91 89 81 79 78 70 58 49 46 42 42 49 65 77 84 84 81 79 79 77 79 81 90 109 125 136 150 146 89 46 45 72 109 105 91 113 110 102 103 102 101 101 102 102 145 146 144 140 133 128 102 57 56 69 78 91 133 174 184 177 168 163 167 177 168 140 113 93 80 73 66 59 52 47 41 41 45 48 47 43 45 48 56 63 70 74 81 87 86 86 89 90 88 84 83 80 82 81 80 76 69 63 54 46 46 45 45 49 61 72 75 69 69 80 93 86 82 89 89 98 112 126 144 147 97 50 58 94 112 61 43 92 111 102 101 99 103 105 102 100 142 144 152 151 121 91 87 59 57 75 80 90 134 176 183 172 159 150 153 149 128 110 95 85 78 70 67 62 56 56 56 53 46 39 32 27 26 36 46 54 61 71 81 84 84 85 88 89 88 89 89 87 86 82 78 70 58 51 44 41 30 26 28 28 33 38 43 48 53 55 59 59 49 57 84 98 107 122 143 148 102 68 79 109 106 53 34 81 111 107 102 102 104 103 102 101 140 145 172 192 173 116 67 45 62 87 86 90 138 177 180 172 157 143 136 130 116 101 92 91 93 91 78 68 60 53 49 46 46 39 29 24 20 20 29 36 46 61 76 84 82 85 90 94 101 106 108 104 98 88 74 58 44 42 37 25 22 28 28 25 31 33 34 42 52 61 73 77 52 28 46 80 102 120 139 146 101 70 95 124 114 61 37 75 116 109 106 104 103 103 105 107 138 151 158 153 175 161 105 64 69 91 91 94 141 176 177 170 161 148 138 128 115 112 120 111 88 67 57 51 51 53 52 49 47 44 34 26 24 26 23 20 26 40 59 76 96 111 109 105 106 108 114 109 94 81 66 46 38 38 28 19 24 20 11 3 1 0 0 0 1 6 29 71 90 65 48 56 86 119 141 147 102 72 103 127 121 73 40 67 109 107 104 106 106 108 109 110 140 154 120 60 111 173 155 117 104 102 91 95 139 175 175 162 154 143 129 115 113 122 112 95 89 84 70 56 43 31 24 19 10 20 28 26 25 22 22 19 15 20 39 69 109 144 140 117 106 107 107 103 94 80 56 40 35 30 28 23 21 44 40 12 2 7 35 35 13 30 22 6 25 57 66 66 91 121 144 147 107 90 113 120 121 81 43 70 103 103 101 101 100 102 103 99 119 138 103 55 108 177 167 136 128 128 108 98 135 171 172 163 152 143 134 124 111 92 87 111 136 100 45 14 2 0 0 0 24 27 18 26 27 32 33 30 22 13 24 54 98 147 155 128 109 103 101 95 88 75 66 57 51 45 31 28 54 85 51 5 8 11 9 6 14 80 107 64 30 54 79 86 97 121 145 149 132 121 118 117 124 87 44 70 112 107 102 103 105 105 103 103 96 127 99 63 131 180 165 151 143 144 116 97 134 172 174 166 156 142 127 113 100 82 66 69 48 27 51 54 23 7 14 24 41 23 49 80 47 25 32 40 37 31 28 44 84 129 144 128 112 103 103 97 86 79 75 71 69 63 60 55 56 77 60 8 0 1 0 0 30 80 76 74 105 106 101 113 120 123 142 153 139 123 120 117 126 95 43 79 111 101 95 91 91 93 94 91 132 145 101 68 141 172 142 154 166 165 122 90 130 169 175 169 168 155 131 109 85 60 48 27 53 120 147 109 28 0 3 2 0 1 58 92 71 45 25 25 41 42 36 56 100 132 142 132 113 101 99 97 94 83 72 74 60 45 36 29 29 44 57 31 3 0 0 20 60 65 58 91 137 137 134 150 152 146 155 158 137 107 91 95 122 105 50 82 96 76 71 71 71 69 69 68 189 177 109 75 147 169 123 132 162 170 157 115 133 172 177 175 175 160 143 131 117 116 133 143 121 104 115 120 69 8 0 0 0 21 61 51 27 15 10 22 36 42 45 81 121 140 143 130 111 99 94 97 96 83 75 73 65 51 40 36 32 35 42 47 41 35 38 49 54 70 92 115 132 129 140 160 164 167 179 170 137 83 60 79 114 106 65 88 92 79 78 78 82 81 80 84 180 177 118 79 150 173 136 137 155 163 181 174 168 178 176 176 185 180 168 170 175 171 178 194 172 100 67 70 71 40 15 14 20 39 39 29 19 13 21 29 36 44 76 123 141 148 144 128 115 108 100 108 112 93 80 73 66 49 42 39 37 36 40 46 51 50 58 62 72 93 110 121 119 119 140 158 162 167 180 174 139 72 43 59 100 101 74 98 95 87 85 85 89 90 90 89 164 170 128 86 152 166 125 107 110 142 165 176 176 179 178 178 188 190 189 191 185 180 177 184 175 133 92 66 53 47 38 36 41 41 36 32 26 26 26 28 38 73 123 148 153 156 149 133 112 103 107 116 134 117 93 82 67 52 46 43 41 39 41 53 62 72 84 91 99 98 102 104 107 118 135 153 157 167 180 179 154 90 41 51 89 104 87 100 96 92 92 92 90 91 91 89 171 175 143 108 158 151 100 73 57 84 136 162 172 177 173 180 193 195 191 186 183 180 173 173 168 149 122 98 84 71 60 59 54 49 44 38 35 34 34 45 73 105 133 148 154 158 151 128 105 96 101 116 141 133 103 92 84 72 58 50 51 52 50 55 68 75 81 85 88 92 98 103 108 119 139 153 146 158 182 179 155 105 45 45 82 108 103 103 92 91 92 94 92 91 91 90 182 184 163 126 155 137 84 59 36 42 108 161 173 174 168 177 193 197 194 187 183 184 176 162 150 136 122 109 99 88 77 72 63 57 52 48 46 42 55 80 106 120 136 155 162 164 152 122 100 94 96 110 140 152 123 99 93 89 83 72 64 67 69 72 75 79 84 92 95 103 106 110 116 123 138 143 141 156 180 178 155 115 51 36 71 114 120 106 95 97 95 92 93 93 95 95 118 118 138 133 151 131 76 53 33 34 108 166 173 176 174 181 193 194 190 179 177 181 175 164 148 126 113 106 94 88 79 70 74 71 65 62 55 62 89 113 126 135 152 168 175 171 150 117 96 92 99 115 147 167 158 121 100 101 100 90 79 75 79 83 85 89 97 105 116 118 117 118 123 128 128 133 148 165 177 178 162 138 74 37 68 119 128 110 98 96 94 93 92 92 94 94 49 50 94 137 156 132 77 50 35 79 155 176 173 173 171 181 191 194 188 172 162 174 183 168 148 134 124 113 104 100 95 88 84 83 71 61 64 89 109 122 142 156 171 180 175 166 148 122 101 94 100 125 155 170 174 158 126 115 109 102 97 88 84 88 90 96 105 110 111 114 121 123 126 125 124 134 157 180 189 181 165 147 98 56 88 125 124 99 84 80 80 79 77 76 76 76 56 53 80 136 163 135 78 51 40 108 174 179 176 172 171 179 191 196 193 183 169 169 173 164 150 142 134 125 117 111 104 96 88 79 67 68 87 106 112 125 152 166 179 181 170 159 142 121 104 92 98 121 147 164 170 169 149 126 115 112 111 100 89 90 92 92 102 111 112 118 125 125 124 120 120 131 160 187 194 183 157 138 97 71 108 123 113 92 79 81 79 78 81 84 85 86 61 58 73 128 169 144 81 51 44 112 175 182 176 171 172 181 192 196 197 195 185 168 158 156 151 146 137 135 125 114 104 94 88 84 84 92 103 105 110 129 155 174 183 178 164 154 138 119 103 98 108 126 145 157 166 170 155 128 113 112 112 106 101 95 97 102 103 108 118 123 123 123 121 117 122 132 149 173 182 173 153 126 74 69 111 118 111 99 90 90 85 86 88 87 87 86 64 61 69 116 170 169 108 53 44 125 174 173 173 168 171 184 191 196 199 200 190 176 160 156 147 130 125 128 122 110 98 92 87 91 99 105 108 107 113 134 160 175 182 173 157 143 127 108 102 101 111 130 149 158 162 166 157 140 124 115 110 111 111 108 104 100 102 108 112 113 117 121 119 124 132 137 136 146 157 156 147 108 51 57 101 127 114 81 72 70 68 65 64 63 59 58 65 65 64 103 151 168 147 107 74 125 167 159 170 165 168 179 190 194 200 202 197 185 172 166 153 130 120 117 114 100 94 93 99 104 107 105 104 102 108 133 160 172 182 174 157 141 126 113 100 97 106 120 136 147 156 163 159 145 130 119 114 108 108 113 108 105 107 108 108 110 113 117 123 131 134 135 138 146 154 148 138 107 62 75 111 150 110 63 57 60 58 59 59 58 60 62 64 63 62 98 140 150 173 160 101 91 144 157 164 163 162 175 189 193 198 203 200 188 174 169 154 137 126 121 118 111 108 106 105 109 109 106 103 101 111 127 150 172 181 174 162 147 125 111 102 101 102 112 126 139 151 158 148 135 128 129 120 112 109 112 113 111 111 114 117 117 118 126 133 134 130 127 132 141 149 143 131 109 90 98 127 142 103 72 70 69 70 73 72 70 74 75 65 68 63 86 141 149 168 155 99 64 101 154 168 164 161 169 183 190 190 192 190 186 178 172 159 141 136 131 124 120 117 114 108 108 107 106 107 105 109 119 140 170 180 173 162 147 126 109 99 99 106 105 115 138 148 151 139 123 120 120 116 116 113 107 106 113 114 116 123 127 129 132 139 138 127 122 130 134 135 129 113 105 103 102 125 135 97 78 75 71 74 76 76 76 78 76 64 64 63 72 126 162 177 156 105 73 77 139 167 162 161 166 176 184 182 176 176 181 182 175 165 153 146 142 130 124 120 116 115 108 105 106 104 97 92 103 137 163 165 169 163 146 124 109 102 102 108 109 116 136 150 139 130 113 106 111 117 116 113 112 118 122 123 127 123 120 130 139 134 127 123 122 122 123 126 120 111 107 105 112 132 130 89 76 76 77 76 76 77 75 76 79 62 61 62 61 101 162 187 169 128 97 93 137 169 164 163 164 170 178 177 168 168 167 166 168 168 161 151 142 135 134 127 121 113 105 102 103 95 84 77 99 136 135 145 170 176 164 142 118 104 101 108 117 119 142 161 138 115 101 95 102 115 117 119 121 122 125 130 129 125 128 132 131 124 117 115 119 120 120 122 115 132 143 132 121 119 115 88 78 79 80 78 77 77 75 75 76 60 61 62 59 77 144 184 170 133 115 125 156 163 166 167 166 166 172 170 166 163 156 155 159 158 155 151 158 158 149 139 128 115 111 109 99 88 76 63 93 124 134 159 179 190 178 143 122 110 106 108 114 128 153 179 159 127 109 97 93 103 111 118 123 124 131 128 129 134 130 126 119 116 114 113 114 114 115 115 98 109 132 121 117 108 128 98 80 79 78 78 78 78 76 76 77 64 62 62 62 67 112 174 177 162 151 141 149 159 165 166 165 166 170 168 163 156 151 150 154 153 152 156 169 171 159 147 135 123 114 103 91 83 66 66 111 141 155 173 188 198 190 156 125 112 112 116 127 139 155 174 171 154 137 123 97 96 106 114 120 123 123 127 129 129 128 128 122 115 113 115 111 106 111 115 81 78 102 111 167 158 121 88 81 79 78 78 77 76 76 79 79 60 61 64 65 62 82 150 181 180 157 151 170 167 154 164 171 170 169 165 159 153 148 146 147 149 155 160 169 169 164 153 139 129 115 98 85 75 59 88 154 174 179 179 182 194 185 149 126 116 118 127 138 143 136 138 148 155 159 168 133 97 98 105 113 116 119 126 127 126 128 126 121 114 111 110 106 106 114 113 77 62 90 109 144 130 85 79 80 80 77 77 78 77 78 79 80 61 61 63 69 66 68 117 177 175 124 125 165 133 111 149 173 174 170 164 156 151 143 142 144 143 146 155 163 170 165 153 143 129 108 95 79 64 53 109 181 191 184 171 163 164 153 131 115 109 113 121 126 122 117 115 118 124 134 149 144 104 94 102 106 110 121 125 121 121 123 123 123 114 108 103 102 108 120 119 76 53 75 98 103 87 78 77 78 79 79 79 78 80 79 78 80 59 56 60 64 66 66 80 137 163 152 141 131 90 71 127 166 176 171 163 155 150 145 141 143 142 142 150 161 168 170 157 139 120 101 87 73 54 47 100 152 148 131 113 106 130 141 126 108 97 94 102 112 109 82 52 49 63 85 106 118 104 97 96 103 112 117 122 126 124 127 124 118 112 106 104 105 110 120 112 74 49 54 71 86 81 79 78 79 78 76 78 79 79 76 73 74 57 59 59 60 59 63 60 101 166 165 136 95 65 63 108 157 175 176 168 161 156 145 135 134 136 139 146 157 169 172 161 139 116 100 87 72 56 49 79 114 101 63 25 2 41 115 120 93 75 69 87 106 88 25 0 5 18 48 79 100 102 95 95 101 107 113 122 132 130 122 119 115 110 108 107 107 116 120 104 70 51 47 59 83 81 78 80 77 76 75 74 79 78 77 76 78 56 59 61 62 60 62 57 88 135 114 95 81 75 69 92 140 163 171 169 162 154 144 135 131 132 135 145 156 167 174 159 133 112 98 84 67 57 64 74 90 81 55 41 40 66 114 116 82 55 54 74 92 92 75 65 63 50 53 72 95 103 99 94 97 104 110 117 123 124 126 127 122 117 114 111 111 114 113 100 68 54 51 60 80 83 80 78 77 78 76 76 76 76 76 76 77 61 59 61 64 64 66 65 66 69 64 65 68 69 66 90 137 158 165 169 161 150 141 137 138 138 141 151 160 169 169 150 126 106 91 80 69 68 78 89 108 131 116 96 115 145 149 129 100 79 73 82 88 96 99 95 85 74 81 95 101 101 101 96 97 102 106 114 121 124 129 130 126 120 115 112 110 114 120 105 69 57 54 60 82 84 79 78 76 73 76 76 76 78 77 75 76 59 58 61 64 64 67 69 67 62 61 63 62 64 68 85 132 159 168 167 160 154 150 143 139 140 145 153 161 170 164 146 130 112 90 77 73 87 107 121 131 153 156 133 133 144 140 115 90 73 62 73 84 92 97 104 103 98 104 108 102 104 104 101 96 98 103 110 118 124 130 132 126 119 111 110 109 112 117 94 63 59 55 61 83 82 77 75 75 77 78 75 74 75 77 78 77 61 58 64 66 64 66 67 67 64 63 66 67 69 67 80 132 165 168 164 161 157 150 142 138 138 147 160 170 176 169 152 132 110 98 89 96 116 131 140 141 146 156 154 147 141 133 111 89 73 59 67 83 92 99 107 116 116 114 109 108 107 108 107 102 101 108 115 121 128 132 127 125 118 111 107 108 115 117 88 61 55 54 62 84 86 79 79 78 76 77 75 75 76 77 74 72 64 61 63 66 66 65 65 67 67 66 68 65 69 70 75 120 159 167 163 164 160 151 143 138 136 140 157 173 176 169 157 139 117 105 96 121 144 146 154 150 146 151 158 149 136 132 110 85 78 68 69 84 96 103 108 118 118 113 116 113 107 110 116 115 108 106 112 121 127 131 129 123 116 111 109 107 113 117 84 56 49 50 59 82 83 78 79 77 76 75 76 78 76 77 75 74 60 61 63 67 68 70 70 67 66 68 67 66 67 70 73 112 157 175 175 162 151 145 141 140 141 146 161 173 173 168 152 136 124 105 104 137 148 151 158 156 152 153 149 141 132 120 100 90 85 74 81 90 93 100 120 122 119 121 121 114 114 118 119 122 122 118 114 122 128 132 130 123 116 112 110 111 116 112 77 54 50 48 56 83 84 77 77 75 76 79 77 73 74 72 74 74 62 62 61 62 62 62 65 65 65 67 67 66 66 68 65 100 153 173 171 161 150 143 145 144 145 151 161 173 171 160 148 138 123 116 143 165 154 146 148 148 151 155 150 134 125 117 100 93 89 77 80 88 89 95 109 115 113 113 109 114 112 112 117 121 128 124 121 125 128 128 125 117 112 108 107 115 122 104 69 55 50 46 54 80 83 76 76 77 78 76 75 75 75 73 73 75 59 64 66 67 67 64 65 65 63 66 65 66 66 68 65 88 146 170 170 166 158 149 143 140 140 146 159 170 169 162 161 155 140 143 168 168 155 148 151 147 143 147 139 128 121 106 95 91 91 76 67 76 84 79 88 111 116 108 102 107 107 113 115 119 124 127 126 128 132 133 126 115 114 111 105 116 126 110 75 57 54 47 51 73 78 75 76 76 75 77 77 75 74 73 74 76 59 61 65 62 60 59 60 62 64 63 61 61 61 64 61 78 132 166 175 170 165 157 148 145 143 149 158 167 169 165 162 156 148 161 163 147 138 137 141 143 144 137 127 119 106 90 86 90 94 80 76 83 81 77 89 103 105 104 102 99 103 110 106 105 113 131 135 127 132 132 124 116 117 117 111 119 127 109 75 57 53 47 50 75 81 76 77 77 73 76 75 73 73 73 73 74 60 59 60 60 59 61 62 61 61 62 64 63 66 68 64 72 121 165 173 169 170 162 154 152 148 155 164 173 172 171 167 157 158 168 155 129 114 123 135 137 138 134 133 119 104 99 97 96 91 83 79 80 77 79 86 95 97 97 100 95 96 107 108 106 105 125 141 132 128 129 123 116 114 111 112 123 126 99 73 60 53 49 58 78 81 75 73 73 74 75 75 73 73 75 73 73 59 60 64 66 63 65 64 64 66 68 68 69 66 70 70 71 109 153 166 170 165 161 161 155 154 159 167 175 172 166 166 165 164 166 146 121 111 126 135 141 145 140 145 142 126 111 103 101 93 84 77 85 91 94 99 103 111 112 106 106 108 104 107 104 95 111 133 131 128 130 123 118 116 116 119 126 124 92 66 57 53 49 50 77 80 72 72 71 74 74 73 71 74 74 74 71 58 59 61 63 64 65 67 66 67 69 68 69 71 72 71 68 92 145 163 166 166 168 167 163 167 172 177 179 177 173 170 168 166 163 132 112 115 127 148 165 170 168 172 169 151 128 114 106 105 91 82 96 102 109 113 113 116 115 110 118 121 114 102 91 98 105 122 131 133 132 120 120 120 122 123 125 117 87 67 56 52 47 51 75 80 72 73 73 75 72 72 72 71 72 74 71 60 57 61 64 62 66 68 66 68 71 71 70 68 69 72 68 78 131 159 162 169 170 171 169 173 180 181 180 178 174 173 173 174 154 118 109 126 152 172 179 174 164 156 138 112 100 93 86 87 86 83 81 88 102 111 117 119 124 129 132 133 125 107 95 100 105 112 121 126 128 124 121 117 114 119 124 108 80 67 58 54 50 54 73 80 76 73 72 73 72 73 71 72 74 72 70 59 55 58 60 63 65 62 63 63 64 67 66 66 64 67 67 71 111 149 155 166 172 171 170 172 174 177 179 179 176 173 174 165 146 124 118 136 153 155 140 113 89 71 51 40 41 43 45 44 48 48 43 39 45 55 62 72 86 100 108 118 121 108 98 95 96 105 122 126 129 126 121 118 115 120 121 98 77 70 61 59 52 54 74 79 73 73 73 71 73 73 72 73 72 71 71 60 55 55 60 61 64 65 65 65 66 67 67 68 67 67 68 71 104 152 161 161 167 167 170 179 178 182 184 183 180 182 177 162 144 129 119 111 108 98 74 51 33 15 0 0 0 13 30 36 24 8 8 4 0 2 6 13 25 37 56 73 81 80 82 93 99 111 124 134 138 130 122 121 119 114 106 87 77 73 63 55 50 53 73 80 74 74 73 72 73 75 73 70 71 73 74 56 58 57 59 63 64 64 65 69 71 69 68 67 66 67 70 70 107 160 165 165 169 168 172 180 183 182 183 182 182 182 179 164 148 131 114 89 65 50 41 40 36 18 2 2 4 4 9 13 12 11 10 8 6 1 0 1 8 22 30 32 36 52 78 95 105 119 131 138 137 130 123 122 118 115 98 79 78 75 68 63 55 57 76 79 73 71 70 72 71 71 72 69 71 72 74 58 59 58 60 62 61 63 63 65 66 67 66 68 67 65 68 68 99 161 168 157 159 166 167 169 174 178 182 186 184 183 180 168 151 139 134 114 81 59 63 64 51 29 13 13 26 36 55 50 46 56 37 22 13 2 1 4 19 38 50 53 62 78 93 101 108 118 133 143 141 133 122 119 118 101 76 77 81 77 72 64 55 53 70 77 73 71 69 70 71 71 70 71 71 70 69 61 57 58 61 60 63 63 62 64 68 65 63 64 68 70 61 39 56 151 177 153 155 164 166 166 173 179 184 187 182 179 177 164 149 143 139 139 140 112 87 81 59 38 27 19 15 15 32 30 18 19 6 0 0 0 5 16 28 47 72 89 101 101 94 97 105 118 133 140 137 130 124 122 111 47 8 33 52 68 71 63 50 49 68 77 74 73 72 73 72 71 70 70 72 70 70 60 59 59 60 60 62 62 64 66 66 67 72 68 54 29 11 0 26 141 186 165 154 159 165 167 171 176 181 184 182 177 172 159 143 138 142 142 146 135 116 104 77 51 41 34 13 0 2 1 0 0 0 0 3 10 22 30 39 58 85 94 97 97 98 100 108 115 125 128 125 125 122 116 71 9 0 0 0 12 31 49 53 54 70 75 72 73 71 69 71 72 69 72 73 69 68 60 60 60 59 59 59 60 61 64 67 62 44 21 2 0 0 1 18 130 190 177 157 155 162 165 171 173 173 176 176 176 169 155 139 140 149 147 134 128 125 116 101 76 63 52 34 18 11 9 16 18 15 13 19 30 39 46 60 79 83 83 86 98 101 101 106 110 119 122 123 124 119 77 14 0 2 2 1 0 0 3 9 25 53 74 74 74 71 69 69 70 72 70 69 69 68 58 57 57 58 58 59 60 62 58 39 16 1 0 1 2 3 0 15 123 189 187 170 155 159 166 168 169 173 172 169 169 168 157 144 148 153 143 129 116 115 116 113 100 73 55 46 40 38 35 40 44 41 41 39 36 41 54 67 67 74 84 87 93 93 98 107 117 124 123 120 120 103 36 0 2 1 1 1 2 2 1 0 0 5 17 34 55 67 74 76 69 68 69 69 66 66 55 53 55 56 57 49 33 23 7 0 0 1 2 1 1 2 0 11 116 188 190 181 161 150 155 162 165 167 169 168 162 164 158 155 156 149 138 130 119 115 114 101 96 84 62 53 52 49 49 51 47 44 45 43 46 53 60 57 62 74 85 94 97 94 98 108 119 122 120 115 116 115 40 0 2 1 1 1 1 1 1 1 2 0 0 0 1 10 31 55 71 70 70 69 68 70 62 60 49 36 17 7 0 0 0 1 2 1 1 1 1 2 0 8 109 188 190 182 173 159 149 154 162 165 167 164 164 163 155 152 151 148 137 128 118 110 114 108 92 82 66 61 59 53 53 53 55 57 54 53 59 61 57 60 67 75 85 91 91 100 105 108 116 118 116 110 120 129 37 0 2 1 1 1 1 1 1 1 1 1 2 2 1 0 0 3 20 43 59 66 72 72 40 24 7 0 0 0 2 2 1 1 1 1 1 1 1 2 0 6 100 185 190 184 179 178 160 142 155 160 161 161 161 161 156 144 138 136 128 128 125 109 103 106 99 80 71 66 57 56 58 59 56 55 60 58 55 55 61 70 75 78 84 89 95 101 107 111 113 116 111 111 141 128 33 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 0 0 6 24 41 57 1 0 0 1 2 1 1 1 1 1 1 1 1 1 1 1 1 4 91 183 191 183 179 177 167 150 147 157 161 159 155 159 156 144 131 127 127 127 125 109 94 93 91 80 75 66 54 45 48 51 48 44 46 47 50 59 68 74 84 90 94 100 102 103 108 114 118 110 105 130 152 122 25 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 0 0 0 7 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 87 181 189 183 179 169 157 152 144 145 154 156 155 156 153 144 129 127 131 130 127 115 105 98 82 71 68 60 50 39 35 36 40 45 52 62 71 78 79 84 90 100 108 107 104 107 111 112 111 109 123 141 153 111 17 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 83 181 187 185 184 175 155 149 155 146 144 148 151 154 153 148 140 133 138 143 138 129 119 106 90 84 82 75 67 61 59 65 70 73 81 87 90 92 90 89 96 107 116 112 110 109 111 110 106 117 134 142 149 96 10 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 75 179 190 188 187 180 169 160 157 157 144 142 147 148 150 151 148 145 148 151 146 137 122 105 98 100 93 84 84 81 80 82 86 91 90 95 102 103 95 88 95 107 118 122 120 119 112 104 118 131 136 144 147 81 6 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 62 168 189 189 185 179 173 164 160 161 158 147 140 145 145 147 151 153 154 156 153 141 118 104 107 108 108 101 92 91 89 87 94 96 99 101 107 112 112 98 93 110 121 122 122 118 108 115 129 129 136 144 135 58 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 39 150 188 186 181 176 171 161 158 161 161 156 144 139 141 145 149 156 157 156 159 147 122 119 131 138 124 106 100 95 93 95 96 98 103 104 107 114 125 116 106 113 123 127 123 113 108 124 133 133 138 142 118 44 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 16 121 185 187 183 181 173 162 158 158 154 158 162 148 136 140 150 158 162 167 167 158 141 137 147 139 121 107 102 97 97 102 104 99 98 103 109 117 123 122 115 118 127 129 119 104 115 129 131 133 138 136 100 32 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 77 168 186 184 177 171 165 163 158 155 156 160 162 145 130 136 154 167 175 177 168 157 151 147 132 126 118 107 99 99 105 105 101 100 102 109 113 118 120 118 126 127 122 108 104 121 129 133 136 136 122 82 24 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 33 136 181 183 179 173 168 163 159 154 156 156 163 164 143 124 134 161 173 170 168 162 154 149 136 126 117 107 101 100 101 104 104 103 108 113 110 108 115 120 122 121 108 107 125 130 126 130 132 125 107 70 20 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 9 90 169 184 178 174 169 162 158 153 155 154 159 166 159 137 126 139 157 167 163 151 146 145 134 121 112 105 100 103 109 108 104 104 108 111 111 106 108 114 115 109 104 123 131 128 128 128 127 117 98 69 17 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 48 134 174 178 175 169 162 155 149 148 152 155 162 166 159 146 127 139 157 158 145 136 136 130 120 110 98 95 100 104 99 92 89 90 98 104 106 110 110 101 98 119 133 128 124 125 122 118 107 92 65 16 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 21 90 152 171 170 165 160 156 146 145 146 149 161 171 167 157 142 127 130 139 138 132 132 121 108 100 92 92 98 99 89 85 85 84 88 94 98 99 93 89 110 135 131 125 122 122 120 110 97 88 56 12 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## 7053 176 174 174 175 174 174 176 176 175 171 165 157 143 134 134 137 138 137 135 135 134 137 135 128 128 129 122 110 107 112 115 123 134 147 151 151 152 155 157 160 164 166 168 168 167 168 169 172 177 178 180 184 188 192 194 191 189 184 177 171 165 154 146 130 118 111 104 99 99 101 105 112 117 119 119 117 116 116 114 121 135 151 161 166 173 174 174 175 174 173 173 173 173 171 171 172 175 175 174 176 176 174 175 176 174 170 159 149 141 136 136 136 138 143 140 136 136 133 123 125 133 126 114 118 126 133 144 157 166 169 166 167 169 174 178 183 191 194 193 189 190 192 196 199 203 206 206 207 211 212 215 214 210 207 202 196 190 185 180 169 155 142 131 120 106 97 96 100 106 109 110 111 111 109 107 110 126 146 157 166 172 174 173 172 172 171 173 175 172 173 172 172 175 176 175 175 175 176 175 176 173 166 156 149 143 139 138 139 142 140 140 135 125 119 120 130 133 125 126 139 152 164 174 181 180 181 183 186 191 194 199 207 212 212 209 205 207 209 212 217 221 224 225 227 228 227 231 232 231 227 221 216 210 203 198 194 187 178 166 149 131 112 98 96 99 101 103 105 104 105 104 106 121 142 158 166 171 173 175 173 174 173 174 173 172 171 172 172 175 176 176 176 176 177 176 175 172 168 161 152 144 140 138 140 139 139 134 124 115 117 127 132 136 139 150 166 181 187 188 190 191 197 202 206 211 212 217 221 221 223 220 216 219 223 228 233 235 237 239 239 240 242 240 241 242 239 235 232 227 218 211 205 203 199 190 178 163 145 124 105 96 95 96 100 100 100 102 102 113 135 155 166 172 173 174 174 174 174 173 173 173 173 171 171 174 174 175 175 177 175 174 175 173 169 160 150 142 133 132 131 129 127 124 117 112 120 133 143 152 166 179 191 196 196 200 205 209 215 219 224 228 228 231 232 230 229 229 230 234 237 240 243 246 246 247 247 248 250 246 246 247 246 242 241 240 235 227 222 215 207 202 197 186 174 157 135 113 96 88 95 100 100 101 99 105 121 143 161 171 174 175 172 172 172 172 173 172 172 172 171 174 177 177 175 176 175 175 176 172 165 157 147 136 127 121 122 122 118 116 111 113 130 147 161 177 193 197 200 203 209 216 219 223 227 231 235 236 235 237 239 238 236 239 240 243 245 246 248 250 250 250 252 252 251 253 252 252 253 249 247 246 241 239 236 229 219 210 205 201 193 181 164 146 120 97 90 94 98 102 101 101 110 130 152 167 172 172 173 173 174 173 172 172 173 171 172 175 177 177 177 177 175 177 174 170 165 155 142 130 122 118 117 114 112 107 112 129 149 166 185 200 207 204 206 212 223 228 228 231 235 238 240 242 241 241 242 243 243 243 244 246 248 250 252 252 251 252 253 253 253 254 253 252 253 252 251 249 248 247 244 240 231 222 213 206 204 196 183 173 153 125 101 92 91 100 102 99 103 121 147 163 171 172 173 175 174 173 174 173 172 172 172 175 175 176 176 176 177 175 172 168 159 148 138 127 121 121 118 111 105 111 129 151 173 189 200 210 211 210 218 227 232 231 232 236 240 240 242 243 241 243 246 242 245 247 247 249 251 253 252 252 252 254 253 253 254 254 254 254 253 254 253 251 250 250 248 244 238 233 222 212 205 203 197 188 176 159 133 108 90 90 101 102 104 118 141 161 171 175 175 175 174 172 173 173 172 172 173 176 175 176 175 175 176 175 172 167 156 143 133 127 127 125 117 109 114 136 157 177 197 204 208 212 215 222 231 237 236 237 240 238 239 242 243 244 245 245 244 245 248 251 250 250 253 251 252 254 252 253 254 254 254 254 254 254 254 253 252 253 252 250 250 247 244 237 230 220 211 208 207 200 191 183 167 140 107 89 95 103 101 111 137 158 170 172 173 174 173 172 173 174 173 172 173 176 176 176 176 175 176 176 173 168 153 141 132 129 132 124 115 117 138 163 183 203 211 212 213 217 224 230 235 239 239 242 243 242 241 241 243 246 244 245 246 247 250 250 252 252 251 252 252 252 254 254 253 254 254 254 254 254 254 252 253 251 250 252 250 247 244 239 233 225 219 214 211 208 202 195 186 169 140 111 96 100 102 104 123 150 167 171 173 176 174 173 173 174 173 173 172 175 176 176 176 175 174 176 175 165 150 138 134 136 129 118 117 139 167 187 203 215 220 222 223 226 228 234 241 243 245 246 245 243 243 243 244 245 245 244 245 247 250 251 251 252 253 252 250 253 252 252 253 254 253 254 253 253 252 251 251 250 249 249 249 246 243 238 231 228 222 217 213 210 209 204 196 189 169 139 115 105 106 105 112 138 161 169 172 174 173 173 171 172 173 172 174 174 175 176 175 176 176 175 173 160 143 138 143 141 124 117 136 167 191 206 218 221 223 227 231 234 238 242 245 247 248 247 246 246 245 247 248 249 249 247 247 247 250 250 249 250 252 251 252 253 252 252 254 254 254 254 252 251 250 251 251 251 251 251 249 246 244 241 237 230 226 223 218 214 209 206 204 199 188 166 141 122 114 109 107 126 154 169 172 171 171 173 172 172 172 171 172 174 175 176 175 175 175 175 168 149 135 135 141 134 125 133 161 190 204 218 224 226 227 227 236 240 241 245 247 248 248 247 245 245 245 246 247 248 248 249 249 250 249 249 249 251 250 250 252 252 254 254 253 252 254 253 253 253 252 252 253 252 252 252 250 246 244 242 242 238 233 228 224 217 211 209 207 201 194 183 160 142 123 113 106 113 140 163 171 171 172 171 172 172 173 172 171 174 174 176 176 176 174 174 162 139 126 129 135 132 134 155 184 202 214 223 225 227 231 235 241 244 246 247 248 246 246 246 246 246 245 246 249 250 248 250 252 251 251 252 249 250 252 252 253 254 254 253 254 254 253 254 254 253 253 252 251 253 252 251 250 248 247 244 243 243 237 232 227 222 215 212 208 203 198 191 172 153 138 119 110 111 125 150 165 170 171 172 173 172 171 170 170 173 173 175 176 177 173 168 153 134 125 129 134 137 148 174 199 212 220 224 226 227 233 238 243 245 247 246 247 246 246 246 247 248 249 251 252 251 252 251 251 253 253 251 253 253 252 254 254 252 252 254 254 254 254 254 254 254 252 251 254 252 250 252 251 249 248 248 244 241 238 235 229 224 217 213 211 204 197 192 181 161 146 129 114 110 113 129 153 166 170 172 172 172 170 172 173 175 175 174 176 174 169 160 144 132 129 133 141 144 162 189 207 215 219 223 224 227 234 239 241 242 243 247 248 247 248 250 249 249 251 251 252 253 251 251 252 253 253 253 254 253 253 252 252 253 253 254 254 254 253 254 254 252 252 253 252 252 252 251 252 249 248 247 243 242 238 236 230 227 223 215 211 205 198 194 185 166 146 132 116 108 111 116 136 157 169 172 173 173 172 172 171 174 174 175 174 172 167 154 138 132 132 139 144 151 173 198 210 215 218 221 225 231 234 239 241 243 245 248 248 248 249 249 250 251 251 252 252 252 251 252 253 253 254 254 254 254 251 252 253 252 252 254 254 253 254 254 253 253 252 252 253 252 251 251 251 248 246 246 245 242 237 233 229 226 224 216 212 206 199 192 185 171 150 134 118 105 105 110 127 152 167 171 173 170 173 173 171 175 174 175 174 171 166 145 128 128 136 143 147 157 185 205 210 215 218 221 227 233 238 240 245 247 247 250 250 250 250 250 249 250 251 251 251 252 253 253 253 254 254 254 254 253 253 253 253 252 252 253 252 252 254 253 253 254 253 253 253 252 250 250 249 245 245 245 242 238 235 233 226 223 222 218 212 205 200 191 183 170 154 136 120 105 99 101 117 143 163 171 172 172 172 172 171 174 173 174 175 174 160 134 120 125 140 144 145 164 192 206 210 216 219 223 228 234 238 242 245 246 247 249 247 248 250 249 248 251 253 250 251 253 252 253 254 252 252 253 252 252 254 253 252 253 252 252 253 253 252 254 254 253 254 253 252 253 251 249 249 247 246 244 240 239 236 235 230 226 223 218 211 204 198 189 182 172 156 139 123 111 103 100 107 131 157 168 171 171 170 171 170 173 173 173 172 173 154 129 121 128 140 140 148 172 195 206 208 213 217 224 229 235 239 241 246 246 246 247 247 250 249 247 249 250 251 252 251 250 253 254 253 253 253 252 253 253 252 253 254 252 253 254 252 252 254 253 253 254 254 253 254 252 250 252 250 247 247 245 244 241 239 235 234 230 225 219 215 205 196 189 181 174 158 139 123 114 109 100 98 121 149 165 170 172 170 171 172 174 173 173 173 169 151 128 121 131 133 135 154 180 199 209 208 209 214 222 231 236 240 242 244 245 244 247 249 248 250 251 249 251 252 250 251 253 253 254 254 254 253 253 252 252 254 253 252 253 253 252 253 253 253 253 254 253 254 253 252 253 253 250 248 247 247 245 243 243 238 237 235 230 227 222 215 206 196 189 180 173 161 142 124 115 112 103 96 109 139 161 169 171 171 171 172 174 173 173 173 167 147 123 121 128 127 137 165 188 200 205 206 207 213 220 230 237 239 241 245 245 245 248 248 249 251 251 252 251 251 252 253 252 254 254 253 254 254 254 254 254 253 252 252 251 252 253 253 252 251 252 251 252 252 250 252 252 250 247 247 245 244 243 240 239 237 235 233 230 226 221 216 207 197 187 179 173 162 144 124 113 109 104 96 100 126 154 169 171 171 172 170 175 175 173 172 164 136 115 123 126 128 150 174 189 196 200 201 207 214 221 226 235 241 242 245 245 246 248 249 249 250 252 250 249 250 252 251 251 251 251 252 253 253 254 254 252 250 250 250 250 252 252 251 249 248 246 245 248 247 247 249 247 249 248 244 242 239 237 237 237 236 234 231 230 228 222 217 210 198 186 178 173 162 146 126 111 107 104 98 97 116 148 166 171 171 172 171 174 173 173 173 159 130 114 119 123 132 153 172 186 197 200 201 207 217 222 224 236 241 243 246 245 245 246 247 247 250 252 248 249 247 249 250 248 247 247 247 248 249 251 251 250 249 249 251 252 251 251 248 245 243 240 240 240 241 241 242 242 244 242 240 238 236 235 235 235 231 231 228 228 226 223 219 209 198 187 177 170 162 148 125 110 107 104 99 97 118 148 166 171 170 170 169 174 173 173 174 160 130 113 112 118 133 149 171 189 202 202 202 207 215 222 229 237 243 243 244 245 245 246 247 247 249 250 250 248 249 249 248 246 244 241 242 244 245 246 248 248 247 247 250 250 250 250 245 239 235 234 232 229 230 230 229 228 225 221 221 223 226 227 228 225 220 224 223 222 221 219 216 210 199 186 179 173 163 150 128 111 106 107 100 103 127 154 168 168 171 171 170 174 174 173 172 162 135 114 109 119 132 147 169 192 201 203 203 206 215 224 231 238 242 242 240 242 244 244 245 247 248 249 247 246 243 241 240 235 233 234 233 233 237 240 244 244 244 244 245 245 246 245 236 233 225 217 214 211 208 207 203 200 196 194 194 198 203 205 207 209 213 217 218 217 214 211 211 212 202 187 181 176 165 146 129 111 102 103 102 112 138 159 168 170 170 169 170 173 174 173 172 164 140 114 109 115 128 144 165 186 198 205 204 209 215 224 233 240 239 235 233 237 239 238 239 240 237 236 233 227 219 213 209 209 212 214 217 219 224 232 238 240 238 237 238 241 241 238 228 217 208 198 191 186 180 175 169 164 159 159 159 162 170 179 183 188 195 204 209 206 204 204 205 207 201 188 181 179 170 149 130 113 99 100 101 115 145 165 169 169 171 172 170 172 174 174 174 167 141 114 110 113 122 137 159 186 202 207 208 211 217 225 232 233 231 227 227 229 224 223 223 219 211 204 199 191 184 179 177 183 190 194 200 202 206 216 226 231 231 228 232 237 236 231 219 203 194 181 173 169 159 148 139 134 133 128 128 134 142 150 158 166 173 184 192 192 192 193 194 200 198 190 183 181 172 154 131 112 100 96 99 116 147 166 169 168 169 170 170 172 174 173 173 170 143 112 108 112 120 133 157 187 204 210 211 214 220 223 225 224 220 214 214 212 209 210 205 192 180 168 162 155 153 154 155 162 172 180 183 187 193 204 214 222 226 226 227 229 227 221 209 194 180 169 160 153 143 132 122 117 112 110 112 115 116 122 131 142 150 156 164 172 175 176 176 183 192 190 182 180 175 157 133 111 98 93 96 118 149 166 169 169 169 167 169 173 175 175 175 173 144 116 108 105 112 126 154 184 203 209 211 213 218 224 218 212 207 207 201 196 193 187 175 158 149 142 134 129 126 129 137 146 155 162 168 176 184 195 204 219 225 225 226 225 222 217 207 192 175 161 148 138 129 121 114 108 105 104 105 107 109 114 123 136 140 142 149 153 156 160 160 168 180 184 181 180 175 162 138 114 97 91 94 115 148 166 170 169 169 169 170 172 175 173 173 172 147 118 106 101 106 121 152 182 203 207 207 210 216 215 208 204 200 196 189 179 167 151 139 131 126 123 120 115 113 115 122 133 139 148 157 166 177 188 200 214 225 229 229 228 223 216 206 188 167 150 139 128 122 119 115 112 115 113 117 125 132 138 145 151 152 155 155 154 155 156 153 155 167 175 175 177 177 166 142 113 93 88 93 115 147 166 169 170 170 169 170 173 172 172 174 173 150 117 100 94 100 119 153 185 204 208 205 208 208 202 199 198 189 181 174 163 151 143 137 134 130 126 125 120 120 121 126 132 136 142 148 156 167 180 192 208 224 230 234 232 224 212 199 182 161 143 128 124 123 124 122 125 127 131 139 148 155 163 168 172 172 173 169 165 162 160 159 151 156 164 166 171 176 169 144 114 91 85 91 111 143 164 169 169 168 167 168 174 174 173 173 173 152 113 94 91 97 122 158 191 207 208 202 203 199 189 190 188 178 171 167 165 168 168 168 164 160 156 150 146 143 140 143 143 144 146 149 154 163 174 189 208 226 234 239 236 225 211 195 177 155 140 129 127 126 129 131 137 144 151 160 170 181 187 194 197 196 190 184 176 168 164 159 152 153 155 155 161 171 170 149 117 92 84 91 114 145 163 168 169 170 168 168 173 175 175 173 172 150 108 90 90 99 128 167 194 206 205 199 197 187 180 185 186 181 182 185 189 195 194 191 190 186 181 176 167 157 152 150 149 151 153 153 156 162 175 193 214 234 241 243 238 227 212 199 177 153 136 127 127 130 134 140 149 161 172 181 192 204 210 213 214 209 199 190 182 174 163 157 151 152 150 149 154 165 170 156 122 92 83 92 122 151 165 168 168 169 168 168 173 172 172 174 171 152 111 91 90 104 134 171 195 205 204 195 186 178 177 185 190 193 198 203 206 211 210 207 206 203 197 191 182 167 156 152 148 147 148 151 157 167 180 198 224 242 250 249 245 234 222 206 179 152 135 123 125 129 138 150 159 167 176 182 190 196 199 202 203 201 196 189 182 174 161 151 143 146 153 152 153 162 168 156 127 94 86 93 124 154 166 167 167 170 168 167 172 173 172 173 173 154 114 91 89 106 141 175 197 206 201 190 179 178 184 191 194 198 202 207 211 215 216 212 212 209 202 193 181 169 157 149 145 143 141 144 153 165 181 206 232 249 252 252 248 241 229 212 180 144 122 119 124 128 134 140 143 144 144 142 141 139 139 145 152 160 166 169 169 166 159 152 144 143 148 150 154 163 167 159 128 98 87 93 122 154 165 169 168 168 167 168 171 174 175 174 172 158 120 89 89 108 144 179 198 204 199 187 181 186 190 191 195 196 201 206 213 214 211 205 197 188 175 161 151 143 137 134 134 133 134 138 148 159 176 206 234 248 253 254 251 246 235 216 177 139 116 116 119 120 120 121 123 125 118 113 109 100 96 96 99 113 125 134 145 152 158 158 149 144 148 150 154 164 168 160 131 98 86 96 124 153 166 169 167 166 166 166 171 172 172 172 171 159 126 92 89 109 143 179 198 204 200 189 187 191 190 187 191 195 199 203 205 199 185 166 147 129 117 110 107 106 108 112 114 119 125 133 143 156 173 201 232 247 253 253 253 248 240 216 173 132 115 114 114 115 114 116 120 124 121 134 131 106 98 101 101 103 110 112 115 128 145 155 155 147 147 152 156 162 168 159 131 99 86 100 130 155 165 168 167 168 167 165 170 171 173 172 172 164 134 93 88 108 140 179 197 203 200 192 190 188 188 187 191 195 198 193 181 159 137 118 109 101 93 89 94 94 97 105 109 114 121 130 143 159 177 198 225 246 251 253 253 249 238 213 169 131 119 116 116 117 113 114 122 139 140 143 138 108 105 115 113 119 123 112 105 106 117 137 149 148 150 155 159 165 167 157 130 100 87 107 138 158 166 167 167 167 166 164 169 169 171 172 172 171 142 97 89 104 133 172 196 201 199 193 191 187 185 189 193 196 187 168 141 124 125 124 120 116 99 90 101 104 107 114 113 115 126 136 150 166 181 201 229 245 250 252 253 247 237 211 174 141 129 124 124 121 115 112 130 160 155 138 131 117 115 120 118 133 140 123 111 111 107 116 136 147 152 157 161 164 166 155 124 95 91 117 148 163 167 168 168 168 166 165 170 170 171 173 173 171 151 110 92 101 125 163 192 201 200 193 192 193 188 191 193 182 157 131 115 127 160 160 134 122 105 95 111 116 122 130 118 115 129 150 168 180 190 207 229 245 248 250 251 247 232 215 188 160 145 143 138 124 119 120 134 159 167 149 138 129 122 121 132 153 157 145 133 126 121 116 126 143 154 158 163 166 169 153 118 92 94 127 156 166 167 167 167 167 167 166 170 172 172 173 172 170 162 128 97 97 119 157 188 199 202 200 196 194 190 191 180 151 131 121 127 148 180 184 149 126 118 118 122 123 135 141 125 118 135 167 190 198 204 214 231 243 246 250 249 246 233 216 198 181 168 163 155 142 139 142 148 163 175 173 163 150 148 151 165 176 172 166 157 146 139 134 131 141 153 161 168 170 171 154 114 87 98 133 157 166 169 166 166 166 165 165 170 170 171 173 172 170 167 140 106 98 110 144 182 201 203 201 197 196 194 184 157 136 139 150 159 169 186 195 173 147 137 140 139 146 156 154 148 149 163 187 205 210 212 221 234 242 243 246 248 243 231 213 202 198 190 178 175 173 171 171 173 177 185 191 188 181 178 180 182 181 180 177 173 169 164 159 152 153 162 170 173 173 173 155 113 88 101 138 161 165 167 167 166 166 166 165 170 171 171 170 171 173 172 151 116 97 103 137 180 202 203 201 197 194 188 170 150 149 162 176 187 191 190 192 186 179 173 173 176 178 179 178 179 186 196 204 209 216 218 223 231 239 243 247 245 243 230 214 205 203 204 196 190 191 189 188 189 190 191 193 190 185 183 180 181 183 186 189 190 192 196 191 179 174 174 180 181 179 177 158 112 86 106 142 161 167 166 167 167 167 167 166 168 170 170 170 172 173 173 159 124 100 103 130 177 201 203 203 201 196 188 172 165 175 190 201 207 207 201 194 189 188 187 188 190 190 192 193 197 204 208 209 214 221 223 225 231 235 242 245 244 238 229 216 206 205 209 210 204 203 203 203 202 204 203 203 204 205 207 207 207 206 208 210 215 217 217 213 206 195 190 192 192 186 180 161 116 90 113 148 164 167 167 167 167 165 164 165 168 169 171 172 170 171 173 163 131 103 101 123 173 203 205 204 202 200 196 188 190 202 213 220 221 221 218 212 210 207 206 207 206 205 205 209 213 215 216 218 226 229 225 226 230 236 241 243 241 237 231 220 210 205 208 213 216 213 213 213 212 213 217 222 227 228 231 234 235 237 236 237 237 234 227 224 219 211 204 201 202 193 183 169 124 98 126 157 164 165 166 165 166 166 165 164 169 172 173 172 171 171 172 165 136 103 95 112 168 204 206 206 208 209 206 206 215 222 228 234 236 236 238 238 235 232 228 226 224 222 216 218 221 225 228 233 235 228 224 226 233 238 242 243 241 241 235 227 214 206 207 212 219 224 224 223 217 213 221 229 236 240 241 244 249 250 249 248 246 239 235 230 224 218 210 208 206 199 188 174 134 113 138 162 172 176 171 167 164 163 164 163 170 171 171 173 171 170 171 168 143 111 94 109 167 204 206 209 216 220 219 225 233 235 238 246 248 248 249 250 247 244 241 237 236 231 224 223 229 236 241 240 236 228 224 228 236 241 245 246 245 243 242 233 220 209 207 212 218 226 233 234 226 217 219 227 234 242 247 249 251 252 250 249 248 244 241 234 228 223 218 214 210 205 193 178 153 146 162 178 186 188 179 169 162 161 162 162 169 169 169 170 170 171 172 169 151 118 95 114 173 209 209 213 221 227 229 236 241 242 246 249 250 252 254 252 249 247 246 243 237 228 224 231 241 245 244 240 234 227 226 228 235 243 250 251 250 249 248 238 225 214 208 208 213 222 231 239 236 231 227 226 226 233 244 249 252 253 251 248 246 246 243 238 233 229 224 219 213 206 194 182 175 181 189 194 193 187 176 167 163 162 162 161 168 168 167 170 177 183 184 178 162 129 110 138 184 208 211 220 227 230 235 241 245 246 247 250 252 252 252 252 251 251 247 240 226 220 232 246 250 250 246 239 231 227 225 228 238 247 253 254 254 254 252 245 230 216 209 208 209 214 225 236 241 242 239 231 225 225 231 238 245 251 250 248 246 247 245 239 237 234 227 221 214 207 195 182 181 194 200 197 189 170 156 158 162 161 161 159 167 170 168 175 191 200 202 198 185 167 162 179 199 209 212 220 231 238 241 245 249 251 251 252 251 252 253 252 251 246 238 230 225 232 247 253 254 251 248 238 229 226 228 234 244 251 252 254 254 253 249 238 225 213 209 206 206 210 222 231 241 246 248 242 236 229 226 228 236 242 245 243 243 245 244 239 234 232 229 222 217 205 195 183 180 189 198 194 176 144 134 152 159 159 160 159 167 167 169 183 198 202 211 216 213 209 208 209 208 209 211 221 234 241 243 245 250 251 251 252 252 253 253 249 244 236 230 233 240 245 251 254 253 252 244 235 228 226 230 231 245 251 250 249 247 241 232 215 204 202 201 200 201 206 218 230 236 244 247 246 244 241 238 235 233 234 235 233 235 239 238 235 231 227 224 221 217 208 196 182 173 183 192 189 162 129 133 152 158 158 160 160 168 166 170 185 175 170 194 215 224 225 224 217 208 206 209 222 234 240 242 246 250 249 252 252 251 251 249 244 238 235 239 245 251 253 254 254 253 249 243 232 227 225 222 224 236 242 238 229 224 219 210 191 182 179 181 187 193 201 214 224 232 239 246 247 247 246 243 240 237 232 231 231 232 233 234 232 229 225 217 216 215 209 196 184 173 177 184 179 160 141 142 154 159 158 160 159 164 163 170 179 149 134 174 206 219 224 223 213 203 205 210 224 235 241 245 249 251 249 251 252 248 247 246 244 240 238 246 252 254 254 253 254 252 247 239 230 220 213 211 210 215 217 210 201 195 192 186 174 164 160 159 167 180 195 208 220 231 239 245 245 248 247 242 241 239 234 228 225 227 227 228 227 223 220 214 211 212 207 197 182 170 174 180 173 170 161 158 159 159 158 158 158 160 159 167 177 141 136 187 213 212 215 214 208 198 200 210 224 235 243 245 248 251 251 250 250 245 242 246 244 245 248 251 254 254 253 254 253 252 246 235 221 209 200 200 198 193 188 179 170 168 169 165 158 150 144 142 151 161 177 199 216 228 238 241 245 247 245 242 239 239 234 227 224 224 222 221 221 218 218 213 209 208 203 193 181 170 171 171 173 182 182 175 165 158 158 157 157 153 144 157 174 147 159 211 228 210 206 209 204 196 197 209 220 233 242 245 249 251 248 249 251 246 246 247 246 249 254 252 252 253 253 253 253 253 247 230 211 192 179 171 168 167 163 156 146 145 148 145 133 116 107 111 121 139 163 187 210 225 236 242 243 245 246 242 239 239 240 235 230 224 219 217 214 214 215 213 208 206 199 191 178 166 162 165 177 193 197 189 173 160 157 157 156 130 116 139 173 158 179 223 229 206 201 205 199 194 194 200 215 231 241 244 248 250 248 247 248 248 250 252 252 252 254 254 253 253 253 253 253 252 243 224 197 165 130 105 97 105 126 138 135 136 137 126 95 68 60 66 84 118 148 177 203 224 236 241 243 245 246 243 240 240 240 240 237 229 225 221 215 214 215 213 208 205 199 191 178 163 151 147 164 189 198 197 184 165 157 156 155 104 105 148 180 168 201 230 224 209 203 204 200 193 192 194 212 232 241 242 247 250 249 248 250 250 251 252 253 254 253 253 253 253 252 253 252 248 244 220 186 148 104 73 58 61 98 135 138 134 133 117 90 64 53 58 79 111 140 168 200 224 237 240 240 242 242 241 241 239 239 241 240 237 235 231 224 220 218 213 208 205 198 189 176 163 143 129 143 168 187 194 187 169 157 154 154 119 136 174 185 181 221 240 234 224 209 191 191 192 192 199 216 232 240 243 247 249 248 248 253 252 252 254 254 253 253 253 253 254 253 251 250 248 243 218 184 152 120 98 86 97 130 153 148 141 133 122 115 93 75 83 100 119 137 164 203 224 236 241 238 241 243 242 242 242 240 241 244 239 239 238 232 226 220 214 208 203 196 186 173 166 159 141 135 152 174 191 192 172 157 154 154 150 162 189 190 196 230 240 234 220 187 153 173 192 192 203 221 232 242 242 244 248 250 250 252 253 253 253 254 254 252 253 254 253 254 253 251 246 239 221 186 157 142 140 146 155 171 177 164 151 140 135 131 125 116 116 119 124 133 168 207 224 234 240 243 242 244 245 243 242 244 244 240 240 241 238 233 230 219 210 206 200 192 180 167 169 185 168 145 148 165 188 193 177 159 153 153 160 170 197 199 212 229 227 212 186 150 128 162 194 195 205 222 233 240 244 246 249 250 252 254 253 253 254 253 253 254 254 254 254 253 253 251 246 238 222 196 166 158 171 180 181 184 187 179 167 154 144 141 140 138 139 132 126 139 177 209 227 235 240 241 243 245 244 244 246 245 245 242 240 240 240 231 221 214 209 203 195 188 176 163 172 199 187 156 149 162 181 194 183 163 154 154 159 175 203 216 224 218 204 186 163 139 134 177 211 198 203 220 231 239 243 246 251 252 253 254 254 254 254 253 253 253 253 254 254 253 251 250 247 238 222 201 180 175 183 185 184 186 190 190 181 168 155 146 147 146 149 145 142 155 181 211 227 236 241 243 243 244 243 245 243 243 240 239 240 237 236 226 218 212 205 200 195 185 170 158 175 206 197 172 159 163 180 199 189 165 154 153 158 170 203 223 223 205 187 175 159 144 156 203 225 203 200 216 229 239 243 245 248 251 254 254 252 254 253 253 254 253 253 254 253 252 250 249 246 239 221 203 192 185 187 191 193 190 190 188 186 176 161 155 158 160 161 161 163 172 189 209 225 234 240 242 242 241 242 244 244 241 236 234 238 236 232 226 216 212 207 198 191 182 167 155 178 206 205 195 178 172 188 207 194 166 155 153 155 166 197 224 221 196 179 171 160 157 179 218 231 206 198 212 224 235 240 240 243 248 251 254 253 253 254 253 252 253 253 252 253 251 251 250 247 238 221 208 202 197 194 196 199 198 191 184 183 180 175 174 176 177 177 177 181 187 197 211 221 230 237 239 240 241 239 242 244 241 238 237 235 235 229 221 216 212 205 196 187 177 163 156 180 204 212 212 198 185 196 209 194 168 155 154 156 163 191 225 219 194 176 170 169 178 200 225 234 211 194 204 218 231 238 241 244 245 249 252 252 253 253 254 254 253 253 252 251 250 252 249 244 234 220 211 205 202 202 202 203 199 190 182 183 188 188 189 193 194 191 191 193 194 201 210 217 223 228 235 238 238 239 242 242 240 240 239 236 232 227 219 212 209 202 191 184 173 159 156 178 198 207 210 204 195 201 206 191 165 155 153 158 161 187 217 219 197 177 170 177 201 219 227 234 213 194 203 214 227 236 241 243 243 246 250 251 253 254 254 254 253 252 251 251 249 248 244 238 228 218 211 208 208 208 210 210 205 193 184 186 191 196 201 203 203 201 196 197 199 201 203 209 214 216 223 230 234 237 240 242 239 235 235 234 231 226 218 209 203 197 189 180 167 158 155 172 188 196 203 203 202 204 202 184 164 155 153 156 162 177 210 220 200 181 171 185 213 230 230 232 216 194 201 215 222 230 238 241 241 244 249 251 252 252 252 252 250 249 251 250 246 243 238 229 223 217 212 212 214 215 216 216 207 194 186 190 194 200 206 208 208 204 198 200 201 201 202 203 207 208 210 218 226 235 241 239 236 233 231 229 230 224 215 207 203 195 187 179 165 155 155 164 170 178 187 193 195 197 186 173 160 153 152 158 157 172 199 218 212 190 181 198 219 230 231 231 218 196 197 209 219 224 235 239 240 244 248 249 250 252 250 249 249 248 249 246 242 238 229 224 216 212 214 213 215 217 220 220 211 195 188 192 196 200 210 214 211 204 198 199 201 198 198 202 204 203 202 210 219 231 237 237 235 232 226 226 225 220 212 208 203 196 186 176 164 152 151 153 151 155 165 176 183 185 180 170 159 152 150 156 157 168 192 213 222 213 204 214 228 231 230 227 213 195 196 202 212 223 232 236 238 243 248 248 249 250 247 246 248 250 249 241 234 230 227 225 216 214 215 213 214 217 225 231 223 208 198 201 207 213 219 222 219 206 199 202 203 197 197 200 198 197 197 201 210 217 227 232 230 228 224 223 224 218 210 206 201 193 186 175 161 152 150 145 141 143 153 165 179 188 188 172 158 152 151 152 153 164 192 209 223 224 220 226 229 228 220 208 201 194 191 199 210 219 228 233 235 242 247 248 249 248 244 243 245 244 239 233 227 225 225 223 218 214 212 210 213 219 231 238 231 221 214 214 223 232 232 228 224 209 202 201 200 196 198 198 195 195 193 195 204 210 217 224 224 222 220 221 221 218 213 204 198 190 180 172 160 153 154 154 147 144 150 164 179 189 187 168 153 152 152 146 143 154 184 210 217 222 223 224 218 210 198 186 188 192 191 196 209 218 224 230 235 241 245 244 243 239 239 243 243 236 229 225 223 222 221 219 214 212 212 212 216 223 224 223 225 223 221 224 228 231 223 212 207 200 198 196 193 192 192 191 192 193 192 192 201 207 211 214 215 217 214 215 216 214 209 202 194 186 175 164 155 153 160 170 171 164 164 173 177 180 177 162 152 152 151 144 143 146 176 206 217 220 225 222 211 198 188 185 188 190 192 194 205 215 224 231 231 234 240 240 237 231 232 240 238 229 220 217 218 219 217 213 209 208 211 215 214 209 199 190 193 201 208 212 207 190 177 170 168 172 180 185 184 185 187 188 187 188 186 188 197 203 206 205 207 209 208 209 207 204 203 199 193 184 174 160 152 154 168 184 186 181 182 179 172 169 161 154 152 151 150 148 146 149 164 193 217 222 226 222 211 203 201 202 204 198 191 192 202 209 218 227 228 228 233 234 228 227 228 230 230 224 214 214 215 210 207 205 202 203 207 205 194 179 164 154 151 157 165 166 156 145 142 140 143 147 152 160 170 174 178 182 180 178 178 181 190 199 202 200 201 201 205 204 201 201 198 193 188 181 169 157 152 157 167 180 182 181 177 168 160 156 153 150 150 149 151 151 150 148 153 176 201 218 222 221 215 214 218 225 220 205 194 192 198 207 215 223 228 227 229 231 226 227 226 221 222 218 211 211 208 200 198 198 197 196 193 182 166 153 147 144 138 139 141 142 144 141 140 141 141 137 136 138 146 154 161 167 170 167 166 173 180 190 194 195 193 197 203 204 204 199 195 192 185 175 162 152 154 160 159 159 159 159 160 158 154 153 150 149 149 149 149 151 151 150 152 158 178 194 201 210 215 220 226 227 223 207 193 190 195 204 213 220 223 225 227 229 229 229 227 221 215 212 207 200 194 189 187 188 188 183 169 155 148 147 148 146 142 140 146 152 148 139 133 132 129 131 130 122 122 128 135 146 153 155 154 159 168 181 186 188 187 191 198 204 203 197 193 186 178 169 158 151 154 158 152 146 146 148 149 152 152 149 150 152 150 149 149 152 153 152 151 152 159 168 176 184 194 204 208 211 211 201 195 189 191 199 208 215 219 223 226 227 227 230 226 219 215 207 196 185 177 173 174 174 167 154 141 134 138 143 143 144 143 141 143 141 136 141 144 147 146 142 134 124 112 107 107 114 124 130 135 144 155 167 177 183 188 192 198 203 202 195 189 184 173 162 155 152 154 156 151 148 148 149 150 151 150 150 149 148 150 150 149 153 153 152 154 153 153 156 159 162 170 174 179 183 182 187 192 192 190 196 204 209 216 222 226 230 228 226 223 217 213 199 184 167 152 147 144 142 133 122 126 138 154 164 170 179 185 188 183 175 177 189 199 204 197 187 178 165 149 126 107 96 93 95 105 119 136 150 165 178 183 188 196 202 201 193 186 178 167 160 153 153 157 156 151 149 150 149 149 149 150 148 149 147 148 150 150 153 152 152 154 154 152 154 155 155 155 158 161 159 162 175 191 196 194 195 199 204 209 214 223 228 231 227 220 212 203 188 168 146 130 117 114 116 118 132 152 177 195 204 209 214 219 220 217 214 213 214 214 211 204 198 195 190 183 168 150 132 113 97 96 101 115 134 150 163 173 181 194 202 199 188 182 172 162 155 152 154 157 154 149 149 149 149 150 149 149 148 148 148 148 148 148 152 153 152 153 154 153 152 153 154 153 155 154 153 156 172 192 198 193 191 196 201 204 209 217 227 232 229 216 206 193 174 152 130 118 129 142 151 157 169 184 198 206 209 211 211 212 209 206 205 205 202 196 194 191 189 193 193 189 183 176 171 163 152 138 122 116 125 141 153 168 182 196 203 197 186 178 168 155 151 149 153 157 155 149 148 150 150 149 149 148 149 149 147 147 148 148 153 154 153 151 152 154 154 152 154 154 155 155 155 154 169 191 196 192 192 195 197 199 205 213 224 230 228 219 207 193 174 154 139 145 173 190 196 198 196 197 200 202 203 206 202 201 194 189 189 187 183 179 178 180 185 188 188 185 182 180 180 180 178 175 161 146 143 153 167 179 192 200 202 195 184 171 160 153 150 149 153 154 154 152 149 149 148 150 150 148 148 147 146 147 148 149 152 153 152 153 153 151 152 154 154 154 154 154 153 153 164 187 196 195 194 193 193 194 201 211 221 226 225 223 215 203 190 179 177 189 205 214 214 209 204 200 197 196 197 194 190 184 175 171 168 167 162 157 158 161 164 168 172 173 173 177 177 180 184 189 193 186 182 184 190 193 199 202 199 190 177 164 154 149 149 150 153 155 153 150 150 149 148 149 149 149 147 148 148 147 147 149 151 152 152 151 152 152 152 153 152 152 153 154 154 154 164 184 197 198 195 193 192 191 195 206 216 225 225 224 221 215 212 214 220 228 232 228 222 215 209 201 197 193 186 177 169 162 156 147 143 141 136 136 143 145 146 151 157 161 165 170 178 186 190 197 207 214 214 209 206 204 202 197 192 180 168 157 149 147 149 150 154 158 155 149 149 149 149 149 149 147 149 147 146 147 148 148 149 152 152 152 151 152 152 153 152 151 152 152 153 152 161 181 195 198 194 194 194 193 193 199 210 220 222 222 221 223 231 238 242 246 247 239 225 216 212 206 197 189 182 169 159 153 147 139 135 130 129 134 139 143 142 143 149 154 160 168 180 188 195 203 213 220 221 216 207 206 198 189 179 172 162 151 147 145 147 151 154 155 153 148 148 148 147 149 148 146 147 147 146 146 147 147 150 151 150 151 151 151 151 152 151 152 153 152 151 153 157 178 195 198 194 194 194 196 192 193 203 214 216 215 216 225 235 244 247 248 251 245 231 222 219 211 201 190 179 168 158 151 147 145 140 138 136 135 141 147 146 147 148 154 162 171 181 192 200 206 213 216 218 208 202 198 193 184 171 163 155 149 145 147 147 149 153 154 150 148 148 147 146 147 149 147 145 146 147 148 145 146 151 151 152 151 151 151 152 152 152 152 153 152 153 151 156 174 191 196 194 193 195 197 194 192 195 205 212 210 212 221 231 240 244 246 250 247 240 233 226 215 205 193 184 174 163 156 153 150 149 146 140 139 145 148 153 157 159 163 170 176 183 193 201 206 207 210 212 207 201 197 187 175 164 155 147 146 148 147 145 149 153 149 146 146 146 148 148 146 146 147 147 146 146 147 146 145 151 151 151 151 151 152 151 150 150 151 152 150 151 153 154 170 188 196 196 193 195 198 196 192 192 198 204 209 211 216 224 234 240 241 245 244 239 234 229 220 210 201 191 181 171 164 159 158 158 154 150 151 153 158 164 169 173 174 178 181 185 193 199 201 203 207 205 206 202 195 184 169 159 151 143 144 146 146 146 148 148 145 145 147 146 148 147 147 147 146 147 145 145 146 145 145 151 152 151 151 152 152 153 151 150 152 152 151 152 151 152 165 184 193 196 194 195 197 196 193 191 193 199 205 206 209 218 229 237 238 239 239 234 229 227 222 215 206 197 188 181 174 171 174 177 175 177 178 175 176 179 184 185 185 184 184 187 190 192 198 202 207 207 204 200 195 182 166 154 144 143 147 146 143 144 146 145 142 142 147 147 146 145 146 145 145 146 145 144 144 143 144 150 151 151 150 151 150 151 152 149 150 151 152 151 151 151 157 178 191 191 191 194 195 194 192 189 189 194 199 199 203 216 226 232 233 234 232 226 222 221 219 216 207 202 195 190 190 193 201 204 205 207 205 199 199 194 188 188 187 187 186 189 190 192 199 207 213 212 205 196 187 176 162 148 144 144 146 146 140 142 146 142 141 143 146 147 144 146 144 145 146 145 143 144 145 144 145 150 150 149 149 150 151 151 152 151 151 150 151 152 152 151 153 168 185 191 189 190 193 192 192 191 189 189 195 197 201 211 219 225 227 228 224 219 216 215 214 215 214 209 207 204 207 215 220 221 221 222 219 217 215 208 198 194 193 193 193 194 196 202 205 211 210 208 200 190 180 170 158 148 144 145 145 142 140 143 144 138 140 143 146 146 146 145 145 146 146 144 144 144 144 144 144 150 148 149 149 149 151 150 149 151 150 151 149 149 149 150 151 162 177 184 186 188 187 190 193 194 190 187 190 198 202 208 212 218 224 223 219 217 216 214 215 217 219 217 218 221 225 227 225 224 223 220 219 220 218 212 207 206 206 205 205 204 205 209 210 207 203 198 194 187 178 167 153 145 143 144 141 140 142 143 139 136 140 143 145 146 147 146 146 145 145 146 145 144 145 144 144 149 149 150 149 149 149 150 148 149 150 149 149 150 149 149 150 156 169 180 184 183 184 186 190 194 191 187 189 197 205 207 211 218 223 221 220 218 220 219 216 219 225 230 230 230 230 230 229 225 219 214 215 215 216 211 209 210 210 211 213 214 215 214 209 204 199 197 192 186 175 162 150 143 143 143 140 141 144 143 136 136 142 145 146 145 146 146 144 144 145 144 144 144 142 144 143 150 149 149 150 148 150 150 148 148 149 150 149 149 149 149 149 152 162 174 181 183 182 183 188 192 192 190 190 195 203 207 211 217 222 227 226 223 224 224 224 226 231 238 236 229 226 227 226 220 211 211 211 211 209 210 210 209 207 210 217 220 216 213 209 203 199 195 185 178 171 160 147 143 142 142 141 142 144 140 133 138 144 146 145 144 145 145 144 143 146 145 143 145 143 143 143 147 148 149 149 149 149 148 147 149 150 147 148 149 148 149 149 150 153 166 177 182 182 182 186 189 191 193 192 190 195 204 209 216 222 222 221 223 225 231 232 230 235 237 238 231 225 224 221 213 207 205 204 206 208 210 212 212 205 206 212 215 212 210 207 203 195 186 177 170 161 152 143 140 140 141 140 144 144 133 133 142 146 145 144 145 146 143 144 143 142 143 143 143 143 143 142 146 147 148 147 148 149 148 150 148 148 147 148 149 147 147 149 147 148 156 168 175 181 182 184 186 188 190 189 188 190 196 202 208 213 217 217 222 229 232 231 230 234 231 232 234 231 226 222 210 201 200 201 204 208 209 209 206 199 200 203 208 209 206 202 195 189 182 171 162 153 144 141 141 142 142 145 142 136 133 137 143 145 144 145 145 146 145 143 144 144 142 143 144 142 143 142 148 147 148 148 147 148 147 148 147 147 148 147 148 147 147 147 149 147 151 163 168 174 180 184 185 186 187 188 183 183 187 193 199 208 213 217 224 227 226 226 229 228 224 225 230 233 229 219 207 197 197 201 204 206 206 201 192 188 191 194 202 206 205 197 190 187 178 167 155 145 140 138 141 144 142 142 137 131 133 141 144 144 145 145 144 143 142 143 143 144 144 144 142 142 143 141 145 145 147 146 146 146 146 145 146 147 147 147 147 148 148 148 147 145 150 165 171 169 177 185 184 184 188 188 183 180 183 187 193 200 209 217 223 227 225 222 223 222 219 223 227 227 222 215 206 200 199 202 204 203 199 194 187 183 183 187 194 202 201 194 189 183 174 160 147 138 135 135 138 142 143 140 133 131 139 141 143 144 143 144 145 144 144 143 141 142 143 142 142 142 142 140
## 7054                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             50 47 44 101 144 149 120 58 48 42 35 35 37 39 38 36 34 31 31 32 32 34 34 34 35 33 32 30 31 33 33 31 33 35 35 35 33 34 34 31 34 33 32 35 39 36 39 42 42 42 40 39 38 38 34 33 34 35 38 37 36 37 41 44 48 51 51 52 54 56 56 60 61 61 63 62 67 76 75 79 81 81 62 52 209 255 251 254 254 254 254 254 254 254 254 254 49 49 47 111 153 136 88 51 43 36 38 37 39 38 36 34 32 32 34 32 34 35 34 32 30 32 32 31 31 31 32 33 32 33 37 35 38 39 37 34 33 34 37 39 40 38 42 44 43 42 41 40 40 38 35 37 38 39 41 41 40 38 42 42 44 48 51 52 54 56 57 61 62 63 66 64 64 73 76 78 83 82 89 51 73 216 255 253 254 254 254 254 254 254 254 254 33 32 30 49 80 83 64 51 41 35 36 36 36 35 33 33 35 36 33 35 36 38 37 34 34 36 34 35 32 33 33 32 33 37 39 37 38 36 33 33 34 35 37 41 38 39 44 46 47 45 43 42 43 38 37 41 43 44 44 43 41 40 42 42 44 46 47 49 54 55 56 58 61 62 63 66 68 70 76 77 83 80 83 94 54 68 205 255 253 254 254 254 254 254 254 254 24 22 23 20 34 52 53 47 41 40 38 36 34 33 32 33 35 32 35 38 40 41 38 38 39 38 38 37 36 37 37 36 37 40 42 39 37 36 34 33 37 39 43 42 40 43 48 51 49 45 45 45 46 42 39 42 47 50 49 47 45 45 44 45 49 48 49 51 52 55 60 61 62 61 61 67 69 70 76 78 79 82 80 86 98 60 123 252 254 253 254 254 254 254 254 254 25 29 29 38 55 58 49 40 39 39 37 35 32 30 35 36 34 35 38 37 41 43 42 39 41 42 40 39 37 38 40 40 38 40 42 40 37 34 36 36 39 42 42 40 41 47 48 49 47 48 50 51 47 44 45 45 48 52 54 53 48 46 45 46 50 51 53 54 54 57 58 61 61 63 63 66 69 70 73 75 80 84 85 83 89 78 67 213 255 252 254 254 254 254 254 254 27 30 38 68 75 59 47 40 39 38 36 34 32 37 37 37 37 37 36 39 45 43 41 43 44 44 43 42 42 45 45 43 42 43 41 41 38 39 39 40 42 43 43 40 43 50 52 54 54 51 50 48 49 52 56 50 48 53 56 60 57 52 48 47 50 50 50 54 55 58 60 62 62 64 68 68 66 70 73 74 78 82 84 86 90 101 58 111 247 253 254 254 254 254 254 254 47 43 51 87 76 51 44 39 36 35 35 35 38 37 37 37 38 35 36 42 43 40 41 44 46 45 43 44 46 49 46 45 46 44 44 40 41 43 41 42 45 45 42 45 50 54 58 59 57 55 52 55 58 60 61 62 56 52 58 64 65 61 56 53 49 50 51 51 53 58 60 61 65 66 68 70 66 67 73 76 78 80 82 86 91 99 100 75 180 255 253 254 254 254 254 254 50 48 61 88 66 44 41 40 36 33 36 41 38 36 37 36 37 36 37 42 42 40 42 45 49 46 45 46 49 54 52 49 47 44 45 45 44 42 44 47 45 44 46 53 58 61 61 62 63 64 64 64 66 66 66 66 62 60 60 65 66 67 64 57 54 53 51 51 52 56 58 61 64 67 70 70 70 69 73 78 76 78 82 85 87 93 110 90 117 243 255 253 254 254 254 254 50 51 64 88 59 40 38 35 35 34 37 39 38 37 35 33 38 36 37 42 40 39 42 46 50 51 52 51 54 56 53 53 50 47 45 47 45 43 45 48 47 49 51 56 61 63 66 69 72 71 70 70 69 71 73 68 63 66 65 64 65 65 66 60 55 54 56 52 53 57 59 62 64 67 71 73 72 73 77 80 78 78 80 85 88 94 103 104 85 189 255 253 254 254 254 254 56 51 61 67 45 39 36 31 34 36 37 36 34 32 33 37 40 37 36 39 40 40 43 46 51 53 54 54 54 54 52 50 49 48 49 47 45 46 49 49 53 56 58 64 66 67 71 74 73 74 74 71 71 76 71 66 66 65 68 70 68 66 66 62 58 56 57 54 55 58 62 66 66 68 70 71 74 74 79 82 81 82 80 84 88 90 102 116 117 183 255 253 254 254 254 254 56 58 59 49 41 37 32 33 35 39 38 35 33 35 36 40 40 36 38 41 42 44 44 47 50 53 55 53 55 56 53 54 50 51 50 51 49 51 54 55 58 61 65 68 71 72 72 74 74 75 73 69 67 69 67 66 69 68 68 70 72 69 67 64 60 58 60 60 57 60 64 67 70 72 73 73 75 76 80 81 81 84 83 84 87 90 99 108 119 187 255 254 254 254 254 254 60 63 59 46 37 35 34 35 35 39 42 38 41 41 39 40 38 39 40 42 45 46 44 46 53 57 57 56 56 52 49 49 48 49 52 53 54 53 56 59 61 67 69 71 74 75 78 79 78 75 73 70 69 71 68 67 69 69 69 69 68 67 68 66 61 60 62 64 64 61 65 70 71 74 77 78 78 82 83 83 82 83 87 88 87 92 97 113 120 171 252 254 254 254 254 254 128 110 73 47 37 34 37 39 38 40 43 42 43 36 37 41 37 38 42 48 49 44 49 47 51 58 56 57 57 49 46 49 51 51 52 54 58 57 59 62 61 65 70 71 76 78 80 81 79 78 76 74 72 72 71 68 68 70 72 70 68 64 63 64 61 59 64 70 69 65 65 70 75 76 77 81 85 86 85 87 87 86 89 89 87 93 99 113 120 178 255 254 254 254 254 254 151 145 97 51 41 36 40 44 42 43 43 43 42 37 39 39 38 41 48 52 50 47 52 48 51 59 60 61 58 52 50 50 50 52 56 57 59 59 60 62 64 69 71 73 78 79 78 81 80 78 78 76 73 72 73 70 67 70 72 72 71 67 64 65 63 63 66 69 72 72 69 70 74 76 81 84 86 88 90 90 90 91 88 88 90 93 101 115 123 161 244 254 254 254 254 254 152 156 113 50 40 41 42 44 45 45 46 44 40 38 41 41 44 50 57 56 50 52 55 49 53 59 63 61 54 55 56 52 54 57 56 56 61 62 61 64 66 70 73 75 76 78 80 80 80 81 78 75 77 76 72 69 68 71 75 76 75 73 71 69 68 68 70 72 75 76 77 76 78 77 77 82 85 87 91 93 93 98 97 92 89 90 98 110 113 126 230 255 253 254 254 254 156 152 104 50 45 44 43 47 46 45 46 42 41 42 44 45 50 57 58 58 55 54 50 49 60 64 61 58 61 60 55 58 59 60 60 62 63 65 66 68 72 70 70 76 79 79 78 77 78 80 79 79 80 79 78 75 74 80 81 82 83 79 79 79 76 75 77 76 77 79 82 82 84 84 81 82 83 87 93 98 100 101 102 98 95 92 94 102 96 84 221 255 253 254 254 254 155 144 86 46 49 47 47 46 42 42 44 43 43 46 46 50 59 60 58 59 56 53 53 61 67 66 63 62 67 64 62 65 64 66 65 63 68 68 69 73 74 73 73 80 83 79 80 82 81 81 81 81 82 84 85 83 82 83 85 86 88 86 86 87 83 81 83 84 82 81 85 85 88 92 91 89 88 88 92 99 105 108 109 106 101 98 94 100 105 80 203 255 252 254 254 254 157 133 70 47 52 50 48 46 43 43 44 45 47 51 52 63 71 70 63 55 53 60 67 69 71 68 69 71 73 74 73 71 72 74 73 72 76 75 74 76 78 79 82 81 82 81 83 84 83 85 85 85 88 89 90 92 91 88 91 90 90 94 88 88 91 89 88 91 90 88 87 88 87 92 95 95 96 95 96 98 104 110 115 113 104 99 98 102 100 82 188 255 253 254 254 254 156 112 48 49 53 52 50 48 46 46 48 49 53 56 62 72 75 71 64 63 70 76 72 71 75 74 73 75 76 82 80 79 80 78 78 81 81 82 81 80 82 83 86 86 88 87 86 87 87 88 89 89 89 93 97 96 96 94 93 95 98 97 96 96 96 96 94 93 92 93 92 88 90 92 95 95 97 99 99 102 107 113 116 118 111 104 103 103 102 82 170 255 253 254 254 254 152 83 41 50 52 52 51 50 49 51 54 54 56 62 74 73 70 71 75 78 82 76 76 78 78 78 77 81 82 84 87 85 80 80 85 87 89 89 88 91 88 84 90 91 89 93 92 89 88 92 94 94 92 94 96 95 92 95 95 94 95 94 93 92 95 97 93 92 95 94 94 91 90 92 96 98 99 98 101 107 110 115 119 120 116 107 104 104 100 79 149 254 250 240 247 252 136 60 42 50 54 53 53 55 59 60 60 62 65 75 79 80 82 82 80 81 82 79 77 81 82 83 84 88 91 89 88 87 89 90 89 88 92 93 92 92 92 91 90 92 94 99 95 91 94 95 95 95 97 98 99 98 93 93 93 90 88 91 89 83 86 86 87 90 90 88 87 92 91 93 95 99 102 101 104 109 114 119 121 122 122 111 104 104 108 75 114 247 242 232 242 247 121 49 46 56 56 53 54 61 71 70 69 69 74 81 87 88 87 84 79 82 82 81 81 82 84 87 89 92 93 91 90 90 90 90 91 88 88 93 92 94 97 98 98 96 96 99 100 99 100 99 100 98 98 97 99 95 93 94 90 86 89 90 85 81 83 86 87 87 86 84 84 88 92 90 94 98 99 102 104 107 112 119 123 125 125 112 103 101 107 87 105 235 254 236 235 240 109 51 55 60 56 54 58 66 73 75 78 79 83 89 93 90 87 86 84 83 84 85 83 82 83 86 87 88 87 84 86 85 80 83 85 84 89 92 92 97 99 97 102 103 100 101 100 101 103 100 100 101 100 96 91 94 93 88 85 85 86 86 82 82 83 82 82 82 81 81 79 81 83 87 90 94 94 96 102 109 116 117 121 128 125 112 103 101 105 106 99 214 255 230 228 241 113 63 58 55 57 55 56 65 74 83 82 84 89 93 93 91 89 87 86 86 86 82 83 86 83 81 79 78 76 76 79 80 79 78 75 79 85 90 92 94 99 102 105 104 102 103 100 99 104 103 100 99 97 94 92 93 89 87 88 84 86 84 80 81 81 81 79 76 75 77 78 77 78 83 87 90 92 98 101 107 116 119 124 128 126 113 103 100 111 115 96 190 255 231 229 238 125 71 52 56 59 53 55 67 80 85 83 88 92 91 91 90 90 87 83 84 82 78 76 76 78 77 74 71 69 69 74 77 76 76 75 78 83 88 93 94 98 102 105 103 100 96 99 100 101 101 101 103 100 96 92 88 85 85 89 86 84 82 82 82 81 80 77 76 76 75 74 77 80 83 85 85 90 96 96 100 113 119 123 127 126 114 102 101 112 112 97 192 255 243 233 236 137 72 54 57 53 52 57 70 84 87 89 91 93 94 93 91 90 83 80 78 74 72 69 68 69 68 68 69 68 68 68 68 72 74 75 79 79 82 88 90 95 100 101 97 96 94 94 95 99 98 99 99 98 95 92 86 85 89 90 85 82 82 85 86 84 82 84 86 81 81 81 82 87 89 87 87 84 87 93 98 109 121 123 127 124 112 101 100 111 115 88 178 255 238 220 234 134 69 56 53 47 49 57 67 84 94 91 89 93 97 93 91 87 79 72 70 68 68 64 67 67 64 64 65 65 65 65 65 67 70 69 74 76 80 85 87 91 95 96 94 95 95 92 91 95 97 98 95 92 90 91 93 91 96 92 84 85 86 87 90 91 90 91 90 89 90 91 91 92 96 91 92 91 88 90 95 103 116 124 126 121 109 97 99 110 117 102 142 247 251 231 229 130 69 52 50 47 50 59 64 86 95 94 95 95 96 92 89 84 78 72 68 68 71 69 68 66 64 62 64 63 61 64 65 66 69 71 72 75 75 76 82 86 91 92 92 93 92 92 89 93 97 94 91 89 91 91 90 90 90 87 85 85 87 89 89 92 92 92 96 99 96 95 97 96 96 95 94 97 96 92 93 100 111 122 125 121 106 97 103 113 111 133 186 240 255 243 235 130 61 52 50 48 53 60 62 87 99 99 98 94 94 91 83 75 69 71 76 79 82 87 86 82 77 77 78 77 76 74 75 74 74 75 74 76 76 75 80 85 88 89 93 93 91 87 88 95 99 97 92 89 91 90 86 84 83 82 82 82 85 87 89 91 93 96 99 98 96 99 99 99 97 98 97 97 98 96 94 98 107 120 124 121 106 96 104 117 105 97 229 255 253 254 243 133 66 50 49 47 54 57 64 89 101 99 94 95 94 84 75 68 71 79 82 85 87 92 92 90 90 93 92 90 89 88 88 83 82 80 75 75 77 77 79 83 85 89 93 93 91 90 88 95 99 98 92 90 87 81 77 79 76 75 75 76 79 82 83 86 95 100 101 101 101 102 104 101 99 97 98 97 96 94 95 96 103 116 124 121 103 94 106 120 113 70 201 255 252 253 243 153 76 47 50 46 52 58 62 87 100 99 97 97 89 79 75 71 79 86 88 87 88 89 93 94 95 97 97 96 95 93 90 88 88 85 81 80 76 75 76 78 83 87 89 92 94 92 93 96 96 96 95 89 80 81 80 76 71 72 72 73 75 80 83 86 92 95 100 105 106 104 102 101 98 95 96 100 96 93 96 98 97 110 124 121 100 94 106 120 124 76 144 252 253 254 245 150 100 51 45 45 53 62 55 84 100 99 100 96 85 76 71 78 84 88 87 87 88 87 91 97 100 98 100 98 98 99 96 94 92 85 82 81 81 78 75 76 79 85 85 89 94 93 91 93 92 94 93 89 83 83 84 79 75 74 77 81 80 84 92 93 97 96 93 96 98 99 102 99 94 94 96 98 97 97 98 97 96 103 118 120 100 91 104 117 129 99 113 242 255 253 252 164 125 57 41 44 57 66 50 79 100 97 98 94 81 74 78 84 85 83 85 84 86 91 97 99 96 97 100 102 100 99 96 95 95 88 85 78 79 78 77 78 77 79 83 84 85 87 89 91 94 92 90 85 79 78 82 81 74 72 75 80 82 83 89 93 96 99 96 94 94 89 88 89 91 92 93 95 97 100 98 96 99 106 118 120 97 91 105 113 124 103 129 244 255 253 248 166 137 71 43 44 62 66 54 76 98 100 98 89 77 78 80 81 82 80 82 85 88 91 96 96 95 98 94 93 91 87 84 84 87 88 90 85 79 76 75 79 80 78 77 81 85 86 88 90 91 89 89 84 76 74 77 73 66 64 68 72 77 76 78 85 89 93 94 97 96 91 90 86 86 91 92 93 95 98 98 95 99 107 121 124 99 93 104 102 109 101 118 243 255 253 251 159 144 92 45 43 62 66 56 74 96 99 97 86 80 83 80 80 82 81 83 85 90 91 88 85 84 87 88 85 82 80 76 71 71 71 75 81 80 74 73 80 84 80 77 77 82 85 87 89 89 89 86 85 76 72 75 68 64 65 70 71 71 71 72 77 82 89 94 98 98 91 91 88 82 82 92 94 94 96 95 95 99 109 123 125 103 95 99 97 100 90 100 229 255 252 251 158 156 124 52 41 61 66 54 60 92 98 93 84 85 84 79 79 81 82 86 85 79 73 75 81 84 89 88 87 83 78 75 78 75 73 71 73 73 69 69 78 85 83 77 76 80 84 85 86 86 86 86 85 77 70 71 66 69 74 76 72 68 68 70 71 72 76 81 85 90 92 91 90 89 87 83 86 96 100 96 93 96 107 123 126 106 98 93 89 93 94 110 199 255 253 252 162 165 140 69 41 57 63 48 54 88 95 91 86 83 80 82 82 84 83 75 67 70 75 80 80 87 94 87 84 80 71 69 70 69 70 74 82 83 72 69 78 85 85 79 76 78 81 80 74 77 83 86 87 82 73 70 72 78 74 67 60 58 63 63 52 46 45 47 52 57 67 78 84 86 88 84 82 87 92 96 96 96 104 121 128 113 102 88 77 87 89 121 191 250 255 247 160 159 149 75 38 56 60 48 55 88 97 95 91 87 83 81 81 80 69 60 68 75 75 74 75 79 79 69 56 49 43 57 70 59 52 59 65 71 81 75 77 83 85 78 76 80 79 76 78 81 83 85 87 86 76 71 73 72 64 63 62 61 62 65 56 48 46 43 44 49 58 60 65 71 75 78 81 87 92 92 95 97 107 120 129 121 101 81 77 82 81 127 209 242 250 246 160 159 160 95 42 57 61 52 53 87 98 93 89 88 84 81 79 73 66 71 75 71 68 72 64 50 38 32 31 30 37 66 73 63 59 62 64 66 75 77 77 85 83 76 77 77 74 78 83 85 86 85 84 86 78 71 70 74 76 78 80 78 76 75 77 72 68 68 67 67 70 68 63 65 70 73 78 82 88 89 93 99 106 122 133 123 93 74 74 79 76 135 232 251 247 245 163 161 170 123 45 56 59 51 48 82 99 95 87 85 85 82 78 71 70 70 67 65 61 54 46 38 34 36 41 48 62 69 68 74 76 78 81 77 75 79 82 85 76 74 76 73 73 81 84 83 85 86 84 85 81 74 73 73 76 79 79 78 77 76 76 75 71 74 77 73 73 74 77 80 81 83 85 88 90 90 95 99 108 122 133 127 95 73 76 80 68 154 250 253 250 242 161 161 168 146 62 48 58 48 44 74 100 100 93 88 86 80 77 73 68 60 57 56 51 50 55 58 58 63 68 71 74 72 71 71 73 72 70 68 75 86 89 83 76 76 76 74 74 78 81 81 82 87 87 85 82 77 75 78 74 73 72 73 75 77 75 75 76 77 76 78 81 84 85 87 90 92 96 99 97 96 99 103 109 120 133 128 99 76 78 81 68 176 255 246 246 237 161 161 165 169 76 39 59 46 43 77 102 101 95 93 89 83 76 73 73 69 67 64 62 65 68 70 67 71 73 71 70 70 69 67 68 67 63 74 82 88 87 81 78 77 73 73 77 82 85 87 86 87 87 85 83 80 78 82 80 73 72 74 75 76 76 77 83 86 86 87 88 88 91 91 91 92 95 103 105 102 104 106 108 119 133 128 101 83 80 88 86 205 255 237 230 225 162 161 162 172 92 38 56 47 49 86 104 103 97 91 89 85 80 83 85 84 84 79 76 78 78 76 77 81 78 79 81 73 68 67 69 69 70 80 86 84 83 81 79 76 74 75 76 81 83 84 86 85 84 86 84 84 83 81 83 82 77 74 80 80 79 82 84 86 90 89 89 90 93 93 95 98 100 101 105 106 105 107 109 115 135 129 101 88 85 72 73 219 255 217 217 224 163 162 162 174 124 50 53 47 57 89 104 105 98 96 95 89 88 93 91 88 85 83 84 87 84 87 90 87 83 82 81 73 70 71 74 79 82 85 87 83 81 82 79 73 72 74 78 79 80 84 82 82 85 90 86 82 85 84 84 86 87 83 82 86 87 90 89 90 94 94 95 97 95 95 97 97 99 102 103 105 107 108 110 116 138 134 97 93 90 81 47 172 255 217 220 215 163 164 163 167 148 72 50 46 59 87 104 104 99 99 95 93 94 95 94 88 85 85 85 85 83 84 85 85 84 82 81 77 74 75 85 90 88 87 84 79 81 82 75 73 73 71 75 78 80 85 84 86 88 90 87 85 87 88 86 87 91 92 86 85 88 89 90 91 92 94 98 100 97 98 99 98 98 100 102 106 105 108 112 118 141 143 97 105 100 93 49 130 254 214 210 216 162 163 161 167 158 86 57 49 55 82 103 102 102 100 97 99 94 94 98 93 93 92 89 89 92 91 90 88 84 81 81 81 77 85 95 95 92 90 82 77 80 78 73 72 74 74 76 82 86 86 85 86 85 85 88 84 85 90 89 89 90 94 94 90 88 90 95 94 94 94 101 102 96 97 100 102 101 102 103 104 108 110 110 117 143 154 100 108 116 97 62 77 235 216 206 215 163 164 162 166 144 93 63 48 52 80 101 104 103 97 99 100 96 95 97 93 93 93 91 90 92 92 92 89 85 84 86 86 88 95 99 95 93 90 83 79 79 74 73 72 74 76 79 84 85 86 86 86 85 86 86 87 87 88 88 90 91 94 95 95 90 89 93 94 94 96 98 99 98 97 101 102 101 104 103 103 107 110 112 114 136 163 112 109 116 98 78 55 209 226 215 216 164 163 165 166 132 83 73 63 50 76 99 105 105 99 97 100 96 94 94 94 93 92 92 91 91 89 91 86 82 85 90 94 94 96 97 95 92 85 83 80 80 77 74 72 73 77 81 82 85 87 86 83 85 85 88 86 88 87 87 88 91 93 97 98 97 97 95 93 94 93 93 95 96 97 103 102 98 103 106 104 105 108 110 114 131 161 134 114 115 103 89 55 185 238 217 219 163 162 163 169 129 63 75 85 54 77 98 103 105 101 98 97 97 95 97 95 92 93 94 92 89 86 86 86 89 92 94 96 92 97 95 91 89 82 82 82 78 75 74 75 78 80 84 86 88 87 87 86 85 87 89 85 86 87 89 88 89 93 97 98 98 100 102 100 95 95 93 96 97 97 101 101 100 101 105 103 102 105 105 112 128 154 146 118 112 108 101 55 167 248 221 217 162 161 166 168 109 60 71 86 66 86 98 99 100 102 103 100 98 98 96 93 92 93 91 91 88 86 88 90 92 96 98 93 91 97 92 91 87 83 83 81 72 72 76 80 80 80 85 90 89 91 90 88 87 86 88 90 88 85 86 88 87 93 96 96 96 98 98 99 95 97 97 97 95 95 98 97 97 99 102 99 103 108 107 111 123 143 144 125 106 111 109 62 159 247 218 220 164 162 167 156 92 69 86 97 69 90 101 100 99 100 102 100 99 101 99 97 98 97 96 95 93 89 90 92 93 98 98 90 90 92 89 92 92 88 81 75 66 70 83 83 83 87 91 92 93 96 94 91 92 89 87 90 89 83 79 83 88 94 96 95 97 99 99 99 98 100 100 98 95 96 99 98 97 99 101 102 103 105 107 112 123 140 143 130 105 113 111 65 163 246 218 218 162 161 169 148 89 82 93 95 80 94 100 101 99 101 100 100 102 103 103 98 99 98 96 92 91 91 92 95 99 98 99 99 96 93 92 93 96 89 76 69 65 80 91 85 85 90 92 94 96 95 94 93 91 90 87 83 93 88 75 79 90 94 98 96 98 101 100 100 99 101 100 99 98 97 98 99 97 97 103 106 106 105 108 109 121 141 141 127 110 111 108 66 177 247 223 217 162 162 169 136 89 85 83 88 96 98 99 99 100 101 98 98 99 99 100 100 98 97 96 91 90 94 96 98 97 97 96 95 94 95 96 99 97 83 71 70 74 93 99 85 88 93 96 95 96 100 100 102 94 88 87 88 100 95 77 75 86 96 97 95 97 100 100 98 99 101 101 98 98 99 97 94 98 99 100 99 101 105 106 108 118 137 133 125 108 109 106 76 203 231 216 219 161 163 169 123 78 86 94 100 95 98 96 97 101 99 96 98 99 96 96 96 97 95 96 95 94 95 95 97 98 96 97 96 99 99 98 97 92 76 71 75 79 89 91 83 84 92 101 102 100 100 105 106 92 82 82 87 96 95 84 74 80 93 97 97 97 98 99 96 96 97 99 100 98 99 97 96 97 98 97 94 96 103 104 107 119 136 128 126 104 107 98 102 233 222 213 217 160 165 165 102 81 94 102 97 89 95 95 95 98 98 97 96 95 97 96 95 93 95 96 95 96 96 97 100 98 96 97 99 100 99 98 93 84 73 76 82 79 76 77 81 82 87 95 98 98 97 97 94 86 77 77 77 81 88 89 82 79 89 98 100 99 93 95 101 99 97 98 96 97 97 97 95 95 96 98 98 102 104 106 105 117 133 124 129 108 100 88 115 240 221 216 220 160 169 154 91 88 104 104 93 91 96 96 94 97 96 96 96 94 95 96 95 92 93 93 93 95 95 97 100 98 98 100 101 100 100 100 88 77 72 74 76 74 68 70 77 79 79 84 86 89 91 89 86 80 78 77 73 75 81 89 88 79 84 94 98 102 97 96 100 98 97 99 94 94 95 96 94 96 98 97 99 104 104 103 105 116 131 123 127 114 99 84 118 243 216 215 217 163 171 142 90 95 106 98 93 94 94 95 94 96 95 94 95 95 94 93 95 92 92 91 91 94 95 96 98 99 100 99 100 102 101 95 86 75 70 71 73 65 51 47 66 78 78 80 80 81 83 81 79 77 78 70 60 63 68 80 88 77 82 94 99 100 100 99 97 98 95 97 98 96 95 93 93 94 95 96 98 99 101 102 102 118 128 121 123 121 103 80 111 241 217 215 216 161 171 147 92 104 100 87 96 95 93 93 93 94 95 93 91 92 96 93 93 91 95 91 92 98 94 96 96 95 99 103 101 101 101 94 89 83 69 66 66 57 52 45 47 64 72 74 73 74 76 74 74 70 56 43 41 48 60 73 80 79 86 96 101 100 99 100 97 95 97 99 101 98 92 91 93 91 92 93 95 99 101 100 102 115 121 117 119 121 108 84 106 230 220 216 216 160 173 150 94 100 89 84 100 95 91 94 95 93 93 91 91 92 96 95 93 94 95 91 90 96 94 96 98 99 100 104 103 103 99 95 93 85 73 63 64 63 66 62 54 58 66 66 65 65 66 71 68 59 52 53 60 62 65 73 75 79 87 94 98 98 98 98 98 99 97 99 97 93 91 91 93 92 91 90 95 100 99 102 103 114 118 116 117 123 105 80 92 219 227 215 214 161 168 139 94 95 78 81 106 98 87 92 96 94 91 89 90 93 96 96 95 93 93 90 90 94 95 97 97 100 103 103 103 99 96 93 89 87 82 74 67 63 63 65 66 64 61 62 62 61 63 68 65 64 71 71 71 75 72 69 70 81 88 92 96 99 100 97 98 97 96 98 97 92 91 91 92 90 90 91 93 96 100 101 102 111 115 114 117 118 95 78 63 190 236 212 216 161 169 125 87 85 70 83 106 97 89 93 96 96 93 90 90 92 93 90 93 94 90 87 91 94 96 98 96 97 101 104 106 101 92 91 90 88 84 79 76 71 70 70 68 63 64 69 70 67 71 73 72 70 72 73 70 69 66 68 76 85 92 94 96 99 100 99 99 96 96 99 98 93 90 90 89 89 91 90 91 94 98 99 106 112 113 113 117 113 88 87 76 163 245 220 213 163 166 112 80 74 63 81 105 98 90 91 94 95 95 91 90 90 89 90 92 92 89 88 88 90 95 95 95 97 101 104 105 98 87 88 89 86 83 82 78 73 71 70 72 72 74 75 81 81 80 82 77 72 73 73 69 70 71 73 78 86 92 91 94 97 100 102 98 96 96 98 98 98 94 93 91 89 91 88 91 95 97 101 105 109 113 112 113 107 94 109 101 128 241 227 208 165 157 106 88 77 70 73 97 101 89 90 94 95 93 90 90 89 89 89 90 91 89 86 88 91 94 94 96 99 101 104 101 92 83 84 83 85 84 84 76 71 68 69 73 73 77 77 84 88 86 85 82 77 74 74 70 70 77 76 78 85 91 92 92 95 99 100 99 96 97 97 95 95 94 92 92 89 88 89 91 94 98 103 105 111 112 107 107 106 101 113 127 98 213 240 210 168 152 97 95 103 94 78 96 99 89 90 92 94 95 93 91 90 92 91 91 89 86 87 90 91 94 94 97 99 102 102 94 80 77 83 83 82 82 80 74 71 70 67 68 70 78 83 85 87 86 87 84 78 79 77 70 71 78 79 82 84 87 93 93 92 95 99 99 97 97 96 97 97 96 93 93 90 88 91 93 93 99 103 108 111 113 107 106 107 106 109 116 102 195 255 212 171 141 87 98 104 100 101 103 100 91 90 93 94 96 93 91 89 88 89 92 90 86 89 89 90 94 95 96 96 99 100 85 69 76 80 80 78 77 76 73 71 71 74 74 78 86 86 86 85 81 83 83 81 81 77 74 73 76 80 80 81 87 91 90 92 93 96 97 96 94 96 98 98 95 96 95 92 92 94 92 95 99 103 106 108 108 107 106 103 106 108 111 85 176 255 213 167 132 92 100 102 100 102 101 95 95 92 92 93 95 94 90 88 88 89 90 90 87 88 91 91 93 91 97 99 96 95 79 72 76 79 78 76 75 75 75 68 68 72 77 85 90 88 87 86 80 82 85 82 78 73 76 78 75 78 84 82 83 88 89 90 94 92 93 97 95 92 95 97 95 96 96 94 95 93 91 94 97 102 105 106 106 108 102 96 102 108 117 77 149 254 213 162 124 101 105 102 102 104 99 93 94 94 94 94 94 95 92 90 91 91 89 88 91 91 91 92 91 91 97 100 98 89 78 75 77 77 74 70 72 75 73 73 75 75 77 86 92 87 88 86 84 83 84 84 80 75 76 75 74 77 79 81 82 83 83 89 90 90 91 96 97 95 94 98 98 96 98 96 95 96 93 92 98 103 106 106 106 105 100 84 93 107 118 82 130 252 207 159 120 100 102 102 104 104 95 93 93 91 91 91 92 95 96 92 90 90 90 89 91 92 94 92 90 93 97 98 94 81 74 74 76 77 74 68 66 70 74 79 81 84 85 87 90 87 85 84 81 78 80 88 86 80 77 78 77 80 75 77 80 80 80 84 85 84 89 91 93 96 97 97 97 97 98 96 96 97 97 95 95 101 103 104 107 105 105 99 103 111 116 80 157 255 214 161 124 97 99 99 104 108 102 97 93 91 90 90 93 95 94 94 93 89 90 91 92 95 93 89 89 94 97 98 85 74 76 72 69 71 73 72 70 73 78 78 80 83 83 85 87 84 82 82 77 80 84 85 82 82 83 79 75 77 75 77 79 81 79 81 81 83 85 85 87 93 95 97 97 98 97 97 97 96 97 95 97 102 104 107 109 107 94 93 93 100 95 141 241 253 217 165 146 100 90 97 101 106 105 99 92 90 92 93 95 98 95 96 96 94 91 91 93 95 93 91 93 97 101 92 77 75 75 74 74 73 74 74 77 79 77 80 82 83 86 89 88 85 80 79 81 84 88 88 83 84 84 80 83 81 73 75 78 79 79 78 77 80 81 83 81 83 91 94 96 95 98 98 97 95 95 95 97 103 104 110 105 106 168 178 177 176 204 248 219 238 223 161 164 147 117 99 93 94 94 93 90 89 91 91 95 96 94 92 94 93 94 93 94 94 92 91 93 98 99 84 73 74 75 78 79 74 73 76 77 81 82 82 83 83 81 83 85 87 87 85 87 92 92 90 90 88 86 85 87 83 80 77 78 78 77 77 76 77 79 81 82 82 87 92 95 98 100 99 99 97 93 96 96 100 104 113 77 170 255 255 255 255 255 229 189 232 229 159 160 165 165 156 146 140 121 97 86 86 89 93 94 93 91 95 95 93 94 94 96 95 92 91 93 99 98 77 66 72 77 79 77 74 76 80 78 79 79 75 72 70 71 69 71 78 85 89 87 88 85 79 81 88 88 85 85 84 84 81 80 79 78 78 76 77 76 75 81 80 83 91 92 95 98 98 94 94 93 92 95 100 105 104 80 204 250 232 248 252 253 224 196 216 207 159 160 160 160 163 164 167 155 101 80 87 88 91 96 94 94 98 97 93 94 96 97 96 93 90 92 99 93 70 67 78 79 80 79 77 79 77 74 69 65 61 58 57 60 59 57 60 66 72 70 67 67 69 69 74 79 79 82 82 82 81 80 77 74 73 76 77 76 75 77 78 77 85 92 97 99 98 94 94 92 91 96 99 102 103 123 220 231 182 234 255 255 226 201 207 192 159 161 162 162 161 159 164 158 104 83 89 87 90 95 95 96 98 98 97 98 95 95 94 88 86 92 100 88 71 71 78 80 80 77 77 76 69 64 59 56 50 47 54 57 57 68 63 59 58 56 54 56 58 59 61 64 66 67 69 72 76 78 78 79 78 77 78 80 78 76 79 76 84 91 94 97 98 97 92 91 93 94 98 99 106 144 159 180 190 231 255 255 223 197 208 193 160 160 161 162 161 159 161 160 112 68 86 88 91 93 95 93 92 95 96 95 96 95 92 90 91 94 99 85 73 77 80 80 82 75 73 68 61 59 54 44 38 54 69 73 66 94 91 87 83 68 67 68 70 69 57 51 52 54 59 62 68 73 79 83 81 78 79 82 80 81 81 76 82 89 92 94 96 94 91 92 93 93 97 97 111 159 104 133 202 226 255 255 221 197 207 196 159 160 159 160 159 159 160 163 155 99 79 88 90 93 92 91 89 93 96 95 95 95 94 93 92 93 97 85 78 82 82 81 78 71 65 59 55 53 46 30 36 54 56 59 60 82 96 95 88 77 80 88 90 88 69 40 41 46 49 56 59 64 69 79 78 73 76 79 80 80 78 77 82 87 91 92 95 93 89 90 93 95 98 96 120 156 105 130 203 218 255 255 221 198 207 192 159 159 160 161 159 160 162 162 166 148 102 87 89 93 93 91 90 94 94 91 91 90 91 91 90 91 95 90 80 82 86 86 81 72 61 52 48 44 37 29 43 55 38 48 68 69 67 95 105 58 65 82 79 73 68 56 41 39 37 38 48 54 57 66 71 71 75 78 79 77 77 78 83 89 89 89 94 93 87 87 88 92 96 93 124 158 99 121 200 214 255 254 217 195 207 195 159 159 160 160 161 159 159 159 160 167 142 90 87 94 92 87 91 94 93 91 91 88 88 90 90 91 95 88 83 84 86 87 90 83 75 64 53 47 44 42 45 51 47 47 57 57 50 78 86 49 57 64 56 54 59 53 45 45 38 31 36 44 49 51 57 65 72 74 76 79 80 82 85 89 90 89 92 90 86 89 91 93 94 92 124 150 101 112 193 212 254 254 212 195 209 193 158 160 160 159 160 160 159 160 159 164 162 101 83 93 92 89 90 88 90 90 88 88 88 88 87 89 91 88 86 87 89 88 89 85 82 79 72 65 61 61 61 61 64 60 55 49 49 52 52 55 54 53 51 52 52 50 51 54 53 51 48 47 46 52 60 69 79 80 80 80 81 83 83 85 87 91 91 88 87 89 91 90 93 93 127 145 101 107 191 210 251 255 216 197 206 189 158 160 160 157 158 161 160 161 160 161 168 119 78 88 92 90 87 88 91 90 87 88 87 84 87 90 88 89 88 89 88 86 86 85 84 80 76 73 72 71 71 73 72 69 68 64 63 64 62 67 70 66 63 62 62 63 65 67 67 68 66 66 65 71 77 76 80 84 82 80 81 84 84 86 85 87 91 87 85 86 89 90 93 91 130 140 103 99 182 214 254 253 209 192 208 191 159 159 158 158 158 159 160 160 160 162 169 138 80 86 91 88 88 88 86 88 90 90 87 84 85 88 91 90 89 85 85 86 83 82 83 84 78 74 75 77 75 72 73 69 70 74 74 71 71 76 76 73 73 74 73 74 73 72 73 74 74 77 78 79 79 81 83 85 78 77 81 86 87 87 87 89 88 85 84 86 89 93 92 94 131 132 101 97 179 208 247 254 210 189 202 185 158 157 157 159 160 159 159 158 162 161 166 152 91 80 91 89 88 87 88 89 88 87 85 84 82 85 87 89 86 84 84 84 84 83 84 83 81 76 74 73 74 76 75 74 76 77 77 78 76 76 77 76 79 79 78 77 75 76 78 78 77 80 81 81 82 83 83 79 75 75 77 83 85 85 87 86 86 85 84 83 87 92 90 94 128 126 105 93 180 200 234 255 209 188 202 187 157 157 158 157 158 159 159 161 161 159 164 162 103 77 87 87 85 84 86 89 86 85 85 85 85 84 88 88 83 80 81 81 82 85 82 81 82 79 77 75 74 75 74 72 79 83 82 80 79 81 80 79 82 79 80 76 74 78 79 78 76 78 81 82 84 83 80 76 77 80 80 83 85 83 83 86 83 84 85 85 86 86 87 95 125 116 107 90 172 202 232 255 206 186 202 185 157 156 158 157 158 159 157 158 159 159 159 166 119 78 85 85 83 84 84 87 88 85 85 83 84 86 87 85 81 79 79 80 77 79 80 82 81 80 81 79 74 70 68 71 76 80 81 79 81 82 79 75 79 78 77 73 72 75 74 74 76 77 79 80 81 82 82 76 77 79 78 79 83 82 81 84 84 83 85 85 87 88 86 97 117 113 110 88 170 204 230 255 205 185 199 182 156 156 157 159 160 160 159 157 158 159 157 167 138 82 79 81 78 83 85 84 82 82 84 84 83 87 88 87 81 78 80 79 75 78 83 84 83 82 80 80 79 72 71 71 71 71 72 74 75 74 75 71 70 73 72 70 72 72 74 76 78 78 79 80 80 80 79 77 77 77 77 78 79 78 78 82 84 84 86 83 84 88 87 104 120 115 111 87 164 207 227 255 202 183 196 182 156 157 157 157 157 158 157 159 158 157 157 163 152 87 76 83 79 80 85 84 81 82 82 81 83 85 86 85 80 77 77 76 76 77 82 82 82 82 78 77 78 75 71 69 70 69 66 68 69 69 69 68 69 72 71 72 70 74 76 73 76 77 79 77 76 78 79 80 77 76 80 81 81 79 77 81 83 83 84 84 85 87 86 109 121 118 112 86 163 219 217 251 200 185 193 177 155 155 157 157 157 156 157 156 157 156 156 161 156 102 75 78 79 81 82 85 84 81 78 78 82 83 83 83 80 78 78 77 77 77 80 82 79 80 81 78 80 78 74 71 67 64 64 65 66 66 68 66 66 66 69 68 65 67 75 76 76 76 79 78 75 76 77 76 76 76 78 81 82 79 80 82 83 84 83 83 86 84 92 112 113 108 108 83 157 222 207 247 201 184 193 178 154 155 155 155 157 156 156 157 157 157 156 159 160 106 73 75 78 76 79 82 82 81 79 81 80 81 83 82 79 77 77 74 77 79 80 81 80 79 81 83 83 80 78 75 70 66 65 65 64 63 69 66 63 65 64 64 68 73 74 78 80 78 78 76 75 76 75 74 76 78 74 72 76 75 78 83 81 83 82 84 88 85 98 130 131 129 131 106 166 217 208 246 196 182 190 178 154 156 156 155 156 156 157 157 157 156 155 157 159 113 74 73 76 77 78 79 81 82 81 81 81 79 82 82 79 78 75 72 72 77 78 79 80 81 81 84 83 84 81 78 78 75 70 69 69 68 67 68 72 70 69 71 74 76 79 78 77 77 76 75 76 77 75 75 76 78 74 70 75 78 81 85 85 84 84 86 85 67 152 247 249 247 236 220 212 200 210 247 193 181 189 174 152 154 156 155 155 156 156 157 155 155 157 158 159 113 78 76 72 74 75 78 82 81 81 82 81 80 81 81 83 82 78 77 75 74 74 74 79 83 83 81 84 85 82 80 82 80 78 74 71 71 71 73 73 74 78 77 77 80 80 78 77 76 72 71 74 77 78 76 73 74 75 73 78 79 81 84 84 82 82 87 73 79 219 255 255 249 223 214 210 195 206 247 197 180 189 174 153 154 156 156 156 158 156 156 157 156 155 157 156 107 79 77 72 72 72 74 79 78 81 82 81 82 80 79 79 78 75 73 75 77 76 76 79 80 82 85 85 85 88 84 83 83 81 79 72 70 76 76 72 76 79 77 78 81 80 78 77 74 72 73 77 80 75 71 71 74 77 77 81 83 81 83 83 80 84 87 62 138 250 254 254 247 212 211 209 187 214 246 188 182 186 173 153 154 155 156 155 157 156 156 154 153 154 156 155 107 81 79 76 74 73 72 77 78 76 78 84 85 82 80 78 74 74 72 74 76 75 76 78 80 82 87 83 84 89 86 84 82 81 82 79 76 76 75 74 75 75 75 76 79 79 79 76 75 75 75 75 77 72 72 74 75 77 78 81 83 83 83 82 82 84 71 97 203 250 254 254 252 227 223 211 194 231 248 189 179 187 173 152 153 154 154 155 155 153 153 155 155 156 156 157 107 83 80 77 72 71 73 73 76 76 78 82 84 82 78 75 76 74 69 73 73 74 76 75 80 83 86 85 86 84 83 83 79 77 74 73 74 76 74 77 77 73 76 75 77 79 80 79 79 77 76 78 75 74 76 76 75 79 81 82 83 80 80 82 85 82 84 143 170 221 255 254 255 253 250 243 239 253 245 187 178 182 168 153 152 152 153 154 156 155 152 155 156 153 155 154 107 86 84 80 76 71 70 71 73 77 78 80 80 78 77 78 76 72 69 69 72 72 75 75 79 82 84 84 82 79 77 77 76 72 74 74 76 78 79 77 74 71 73 76 78 78 77 80 78 76 76 77 76 76 76 77 78 78 78 80 83 80 79 81 82 82 118 149 193 245 254 253 254 254 255 255 255 255 240 181 176 178 167 151 152 153 153 153 153 153 156 154 151 152 157 152 109 88 87 84 81 76 68 67 69 72 77 78 79 78 77 78 75 72 72 71 72 74 75 72 76 80 82 81 79 76 76 78 78 76 78 77 78 79 76 76 75 73 74 75 76 75 75 77 76 73 72 72 75 76 73 73 77 77 79 80 79 80 77 78 83 79 140 224 255 255 254 254 254 254 254 253 252 255 239 180 173 178 165 152 152 153 153 152 151 152 154 154 153 151 155 150 108 88 87 84 83 78 73 67 66 70 72 72 74 73 71 76 76 72 73 73 74 73 73 72 77 80 80 78 77 79 75 75 78 79 79 80 79 77 77 76 77 77 75 73 72 75 77 74 74 74 71 69 72 73 71 73 75 75 79 79 77 76 75 80 85 60 178 255 252 254 254 254 254 254 254 254 253 255 236 174 175 175 165
##      .src   ImageId     .rnorm .pos Image.pxl.1.dgt.1 .lcn
## 7050 Test Test#0001  0.1777388 7050                 1     
## 7051 Test Test#0002 -0.5182230 7051                 7     
## 7052 Test Test#0003 -0.5757552 7052                 1     
## 7053 Test Test#0004  0.6224964 7053                 1     
## 7054 Test Test#0005  1.0077015 7054                 5     
##      left_eye_center_x.Final..rcv.glmnet
## 7050                            66.66288
## 7051                            66.71857
## 7052                            66.72326
## 7053                            66.62762
## 7054                            66.59693
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
## Loading required package: tidyr
## 
## Attaching package: 'tidyr'
## 
## The following object is masked from 'package:Matrix':
## 
##     expand
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
## All.X##rcv#glm                 3.843930  0.003175106      0.003192879
## Low.cor.X##rcv#glmnet          3.843935  0.003172451      0.003192721
## All.X##rcv#glmnet              3.843935  0.003172451      0.003192721
## Max.cor.Y.rcv.1X1###glmnet     3.843935  0.003172451      0.003192721
## Max.cor.Y##rcv#rpart           3.850047  0.000000000               NA
## Final##rcv#glmnet                    NA           NA      0.003162455
##                            min.RMSE.fit
## All.X##rcv#glm                 3.268560
## Low.cor.X##rcv#glmnet          3.268553
## All.X##rcv#glmnet              3.268553
## Max.cor.Y.rcv.1X1###glmnet     3.270851
## Max.cor.Y##rcv#rpart           3.266070
## Final##rcv#glmnet              3.441975
```

```
## [1] "All.X##rcv#glmnet OOB RMSE: 3.8439"
```

```
##   err.abs.fit.sum err.abs.OOB.sum err.abs.trn.sum err.abs.new.sum
## 0        78.52908        57.77299        136.1512              NA
## 6       540.63367       256.84422        798.3770              NA
## 3       680.84293       356.53763       1038.7135              NA
## 4       624.38775       306.05286        932.2126              NA
## 1      4125.29454      1685.02074       5817.5716              NA
## 9       519.57464       195.59648        715.2853              NA
## 8       601.16807       173.24281        773.8661              NA
## 5       466.26785       272.12393        738.9476              NA
## 2      2205.95815       884.47657       3086.5713              NA
## 7       590.18984       173.73053        764.0616              NA
##   .freqRatio.Fit .freqRatio.OOB .freqRatio.Tst .n.Fit .n.OOB .n.Tst .n.fit
## 0     0.00514444     0.00906801    0.008412787     26     18     15     26
## 6     0.05203799     0.05491184    0.054402692    263    109     97    263
## 3     0.06390977     0.07707809    0.076836792    323    153    137    323
## 4     0.06509695     0.06649874    0.065619742    329    132    117    329
## 1     0.39572616     0.37481108    0.378014582   2000    744    674   2000
## 9     0.04610210     0.04483627    0.044307347    233     89     79    233
## 8     0.05302731     0.04030227    0.039259675    268     80     70    268
## 5     0.05203799     0.06498741    0.064498037    263    129    115    263
## 2     0.21329640     0.21863980    0.220415031   1078    434    393   1078
## 7     0.05362089     0.04886650    0.048233315    271     97     86    271
##   .n.new .n.trn err.abs.OOB.mean err.abs.fit.mean err.abs.new.mean
## 0     15     44         3.209611         3.020349               NA
## 6     97    372         2.356369         2.055641               NA
## 3    137    476         2.330311         2.107873               NA
## 4    117    461         2.318582         1.897835               NA
## 1    674   2744         2.264813         2.062647               NA
## 9     79    322         2.197713         2.229934               NA
## 8     70    348         2.165535         2.243164               NA
## 5    115    392         2.109488         1.772882               NA
## 2    393   1512         2.037964         2.046343               NA
## 7     86    368         1.791036         2.177822               NA
##   err.abs.trn.mean
## 0         3.094345
## 6         2.146175
## 3         2.182171
## 4         2.022153
## 1         2.120106
## 9         2.221383
## 8         2.223753
## 5         1.885071
## 2         2.041383
## 7         2.076254
##  err.abs.fit.sum  err.abs.OOB.sum  err.abs.trn.sum  err.abs.new.sum 
##      10432.84651       4361.39876      14801.75787               NA 
##   .freqRatio.Fit   .freqRatio.OOB   .freqRatio.Tst           .n.Fit 
##          1.00000          1.00000          1.00000       5054.00000 
##           .n.OOB           .n.Tst           .n.fit           .n.new 
##       1985.00000       1783.00000       5054.00000       1783.00000 
##           .n.trn err.abs.OOB.mean err.abs.fit.mean err.abs.new.mean 
##       7039.00000         22.78142         21.61449               NA 
## err.abs.trn.mean 
##         22.01279
```

```
## [1] "Features Importance for selected models:"
```

```
##      All.X..rcv.glmnet.imp Final..rcv.glmnet.imp
## .pos                   100                   100
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
## 21     predict.data.new         10          0           0 132.313 148.222
## 22 display.session.info         11          0           0 148.222      NA
##    elapsed
## 21  15.909
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
## 1                import.data          1          0           0   8.084
## 21          predict.data.new         10          0           0 132.313
## 20         fit.data.training          9          1           1 122.385
## 16                fit.models          8          1           1  96.119
## 15                fit.models          8          0           0  86.922
## 17                fit.models          8          2           2 105.616
## 2               inspect.data          2          0           0  72.246
## 18                fit.models          8          3           3 113.717
## 19         fit.data.training          9          0           0 118.216
## 13   partition.data.training          6          0           0  81.715
## 14           select.features          7          0           0  84.629
## 3                 scrub.data          2          1           1  78.495
## 10      extract.features.end          3          5           5  80.362
## 11       manage.missing.data          4          0           0  81.265
## 12              cluster.data          5          0           0  81.653
## 9      extract.features.text          3          4           4  80.307
## 6    extract.features.string          3          1           1  80.181
## 4             transform.data          2          2           2  80.118
## 7  extract.features.datetime          3          2           2  80.235
## 8     extract.features.price          3          3           3  80.273
## 5           extract.features          3          0           0  80.160
##        end elapsed duration
## 1   72.245  64.161   64.161
## 21 148.222  15.909   15.909
## 20 132.313   9.928    9.928
## 16 105.615   9.496    9.496
## 15  96.118   9.197    9.196
## 17 113.717   8.101    8.101
## 2   78.494   6.248    6.248
## 18 118.215   4.498    4.498
## 19 122.384   4.168    4.168
## 13  84.628   2.914    2.913
## 14  86.922   2.293    2.293
## 3   80.117   1.622    1.622
## 10  81.265   0.903    0.903
## 11  81.652   0.387    0.387
## 12  81.715   0.062    0.062
## 9   80.362   0.055    0.055
## 6   80.234   0.053    0.053
## 4   80.160   0.042    0.042
## 7   80.273   0.038    0.038
## 8   80.307   0.034    0.034
## 5   80.180   0.020    0.020
## [1] "Total Elapsed Time: 148.222 secs"
```

![](Faces_tmplt_files/figure-html/display.session.info-1.png) 
