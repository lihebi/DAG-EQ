library(bnlearn)
data(learning.test)
str(learning.test)



test_learn <- function() {
    bn.gs <- gs(learning.test)
    bn.gs

    bn2 <- iamb(learning.test)
    bn3 <- fast.iamb(learning.test)
    bn4 <- inter.iamb(learning.test)

    ## FIXME not "TRUE"?
    compare(bn.gs, bn2)
    compare(bn.gs, bn3)
    compare(bn.gs, bn4)

    bn.hc <- hc(learning.test, score = "aic")
    bn.hc

    compare(bn.hc, bn.gs)
}

test_plot <- function() {
    bn.gs <- gs(learning.test)
    bn.hc <- hc(learning.test, score = "aic")
    ## plot using plot
    par(mfrow = c(1,2))
    plot(bn.gs, main = "Constraint-based algorithms", highlight = c("A", "B"))
    plot(bn.hc, main = "Hill-Climbing", highlight = c("A", "B"))
}

test_plot_2 <- function() {
    bn.gs <- gs(learning.test)
    bn.hc <- hc(learning.test, score = "aic")
    ## plot using graphviz
    par(mfrow = c(1,2))
    highlight.opts <- list(nodes = c("A", "B"), arcs = c("A", "B"),
                           col = "red", fill = "grey")
    graphviz.plot(bn.hc, highlight = highlight.opts)
    graphviz.plot(bn.gs, highlight = highlight.opts)
}

test_knowledge <- function() {
    ## background knowledge
    bn.AB <- gs(learning.test, blacklist = c("B", "A"))
    compare(bn.AB, bn.hc)
    score(bn.AB, learning.test, type = "bde")
    bn.BA <- gs(learning.test, blacklist = c("A", "B"))
    score(bn.BA, learning.test, type = "bde")
}

test_modify <- function() {
    bn.gs <- gs(learning.test)
    bn.hc <- hc(learning.test, score = "aic")
    ## text visualizing the graph
    modelstring(bn.hc)
    undirected.arcs(bn.gs)

    ## modify !! the graph
    bn.dag <- set.arc(bn.gs, "A", "B")
    modelstring(bn.dag)
    compare(bn.dag, bn.hc)

    ## violating acyclic constraint will result in error
    set.arc(bn.hc, "E", "A")

    ## create a bn with existing graph
    other <- empty.graph(nodes = nodes(bn.hc))
    arcs(other) <- data.frame(
        from = c("A", "A", "B", "D"),
        to = c("E", "F", "C", "E"))
    other

    ## this is useful to compare the learned structure with (HEBI: ground truth?)
    score(other, data = learning.test, type = "aic")
    score(bn.hc, data = learning.test, type = "aic")
}

test_alarm <- function() {
    ## alarm dataset

    alarm.gs <- gs(alarm)
    alarm.iamb <- iamb(alarm)
    alarm.fast.iamb <- fast.iamb(alarm)
    alarm.inter.iamb <- inter.iamb(alarm)
    alarm.mmpc <- mmpc(alarm)
    alarm.hc <- hc(alarm, score = "bic")


    ## compare with ground truth
    dag <- empty.graph(names(alarm))
    modelstring(dag) <- paste("[HIST|LVF][CVP|LVV][PCWP|LVV][HYP][LVV|HYP:LVF]",
                              "[LVF][STKV|HYP:LVF][ERLO][HRBP|ERLO:HR][HREK|ERCA:HR][ERCA][HRSA|ERCA:HR]",
                              "[ANES][APL][TPR|APL][ECO2|ACO2:VLNG][KINK][MINV|INT:VLNG][FIO2]",
                              "[PVS|FIO2:VALV][SAO2|PVS:SHNT][PAP|PMB][PMB][SHNT|INT:PMB][INT]",
                              "[PRSS|INT:KINK:VTUB][DISC][MVS][VMCH|MVS][VTUB|DISC:VMCH]",
                              "[VLNG|INT:KINK:VTUB][VALV|INT:VLNG][ACO2|VALV][CCHL|ACO2:ANES:SAO2:TPR]",
                              "[HR|CCHL][CO|HR:STKV][BP|CO:TPR]", sep = "")
    alarm.gs <- gs(alarm, test = "x2")

    ## FIXME this is very slow
    ## alarm.mc <- gs(alarm, test = "mc-x2", B = 10000)

    par(mfrow = c(1,2), omi = rep(0, 4), mar = c(1, 0, 1, 0))
    graphviz.plot(dag, highlight = list(arcs = arcs(alarm.gs)))
    ## graphviz.plot(dag, highlight = list(arcs = arcs(alarm.mc)))
}
