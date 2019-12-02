## load the pcalg library
library("pcalg")


test_data <- function() {
    ## list all dataset, we can see pcalg provides many datasets
    data()
    ## load gmG data
    data("gmG")

    ## exploring the data
    ls()
    ## [1] "gmG"  "gmG8"
    names(gmG)
    ## [1] "x" "g"

    gmG$g
    ## A graphNEL graph with directed edges
    ## Number of Nodes = 8 
    ## Number of Edges = 8 
    dim(gmG$x)
    ## [1] 5000    8

    ## the data is plan matrix, with no attributes

    ## The gmG8 seems to be the same data
    gmG8$g
    dim(gmG8$x)

    ## Run some discovery

    ## the covariance
    dim(cor(gmG$x))
    ## [1] 8 8
}


test_PC <- function() {
    data("gmG")
    suffStat <- list(C=cor(gmG$x), n=nrow(gmG$x))
    ## the suffStat actually contains all information needed for CI test
    pc.fit <- pc(suffStat, indepTest=gaussCItest, p=ncol(gmG$x), alpha=0.01)
    stopifnot(require(Rgraphviz))
    par(mfrow=c(1,2))
    ## The g can be plotted!
    plot(gmG$g, main="")
    ## pc fit actually contain some data
    plot(pc.fit, main="")
}

test_ida <- function() {
    # first run PC to get the graph
    data("gmG")
    suffStat <- list(C=cor(gmG$x), n=nrow(gmG$x))
    pc.fit <- pc(suffStat, indepTest=gaussCItest, p=ncol(gmG$x), alpha=0.01)

    ## do some inference

    ## intervention on V1, by increasing the value by 1 (HEBI: seems to be hard
    ## coded), and observe the effect on V6
    ida(1, 6, cov(gmG$x), pc.fit@graph)
    ## [1] 0.7536376 0.5487757
    ##
    ## There are two return values, because the causal structure is not unique
    ## (HEBI: the hidden confounders?)
    ##
    ## FIXME if multiple graphs, plot multiple?

    ## idaFast basically compute the effect of V1 on 4,5,6 simultaneously
    idaFast(1, c(4,5,6), cov(gmG$x), pc.fit@graph)

}



## Testing FCI, which returns a PAG
test_FCI <- function() {
    ## FIXME why gmL dataset?
    data(gmL)
    suffStat <- list(C=cor(gmL$x), n=nrow(gmL$x))
    pag.est <- fci(suffStat, indepTest=gaussCItest, p=ncol(gmL$x), alpha=0.01, labels=as.character(2:5))
    par(mfrow=c(1,2)); plot(gmL$g, main=""); plot(pag.est)
}

test_GES <- function() {
    score <- new("GaussL0penObsScore", gmG8$x)
    ges.fit <- ges(score)
    par(mfrow=1:2); plot(gmG8$g, main = ""); plot(ges.fit$essgraph, main = "")
}

test_RFCI <- function() {
    data("gmL")
    suffStat1 <- list(C = cor(gmL$x), n = nrow(gmL$x))
    pag.est <- rfci(suffStat1, indepTest = gaussCItest,
                    p = ncol(gmL$x), alpha = 0.01, labels = as.character(2:5))
    par(mfrow=c(1,2)); plot(gmL$g, main=""); plot(pag.est)
}

test_GIES <- function() {
    data(gmInt)
    ## inspect the interventional data
    names(gmInt)
    dim(gmInt$x)
    gmInt$g
    gmInt$targets
    ## FIXME all data are interventional?
    length(gmInt$target.index)

    score <- new("GaussL0penIntScore", gmInt$x, targets = gmInt$targets,
                 target.index = gmInt$target.index)
    ## TODO there are a lot of information in the score
    names(score)
    gies.fit <- gies(score)
    ## and simy
    simy.fit <- simy(score)

    par(mfrow = c(1, 3)) ; plot(gmInt$g, main = "")
    plot(gies.fit$essgraph, main = "")
    plot(simy.fit$essgraph, main = "")
}

test_FCIplus <- function() {
    ## FIXME data
    ## FIXME plot
    suffStat <- list(C = cor(gmL$x), n = nrow(gmL$x))
    fciPlus.gmL <- fciPlus(suffStat, indepTest=gaussCItest,
                           alpha = 0.9999, labels = c("2","3","4","5"))
}

test_LiNGRAM <- function() {
    ## for two variables case

    ## generate data
    set.seed(1234)
    n <- 500
    ## Truth: stde[1] = 0.89
    eps1 <- sign(rnorm(n)) * sqrt(abs(rnorm(n)))
    ## Truth: stde[2] = 0.29
    eps2 <- runif(n) - 0.5
    ## Truth: ci[2] = 3, Bpruned[1,1] = Bpruned[2,1] = 0
    x2 <- 3 + eps2
    ## Truth: ci[1] = 7, Bpruned[1,2] = 0.9, Bpruned[2,2] = 0
    x1 <- 0.9*x2 + 7 + eps1
    ## Truth: x1 <- x2

    ## run lingram
    X <- cbind(x1,x2)
    res <- lingam(X)
    res    
}


## TODO test it
test_randDAG <- function() {
    n <- 100; d <- 3; s <- 2
    myWgtFun <- function(m,mu,sd) { rnorm(m,mu,sd) }
    set.seed(42)
    dag1 <- randDAG(n=n, d=d, method = "er", DAG = TRUE)
    dag2 <- randDAG(n=n, d=d, method = "power", DAG = TRUE)

    ## also sample edge weights
    dag3 <- randDAG(n=n, d=d, method = "er", DAG = TRUE,
                    weighted = TRUE, wFUN = list(myWgtFun, 0, s))
    dag4 <- randDAG(n=n, d=d, method = "power", DAG = TRUE,
                    weighted = TRUE, wFUN = list(myWgtFun, 0, s))    
}
