install.packages("bnlearn")
install.packages("Rgraphviz")

library(bnlearn)


data(learning.test)
data(gaussian.test)

learn.net = empty.graph(names(learning.test))
modelstring(learn.net) = "[A][C][F][B|A][D|A:C][E|B:F]"
learn.net

net = hc(learning.test)
## plot the net?
plot(learn.net)

## empty graph
gauss.net = empty.graph(names(gaussian.test))
modelstring(gauss.net) = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"
gauss.net
## scores
score(gauss.net, gaussian.test, type = "bge", iss = 3)

graphviz.plot(gauss.net)

plot(inter.iamb(learning.test))

load("/home/hebi/Downloads/sangiovese.rds")
sangiovese = readRDS("/home/hebi/Downloads/sangiovese.rds")
