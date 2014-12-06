Bayesian Model Averaging(BMA) on Naive Bayes(NB) classifier[PDF:doc/Bayesian Model Averaging/paper]

Brief introduction
In this work, we derived new classfier BMA for NB[scr/classifiers/BMAOfNBWithGaussian.java] and built its baseline NB classfier[scr/classfiier/GaussianNaiveBayes.java]. We expected to design a classifier that use BMA to minimize the risk of signal model uncertainty while still remain better predictive accuracy than signal model(Naive Bayes) with acceptable running time. To evaluate the novel classifier, we use cross-validation. we, then, compared the results with other well known classifiers such as SVM and Logistic Regression. Those classifiers belong to Liblinear package,see[http://www.csie.ntu.edu.tw/~cjlin/liblinear/].The code in this project is based on code of Feature Selection project provided by Rodrigo Santa Cruz[https://code.google.com/p/feature-selection-high-recall/]. More information about this project, please see our paper on Dr.Scott Sanner's web[http://users.cecs.anu.edu.au/~ssanner/].

To see structure of the BMA on NB classifier, ckeck following codes:
scr/classifiers/BMAOfNBWithGaussian.java: classifier
scr/utils/ProbabilityGenerator.java: saving structure shared with normal Naive Bayes Classifier
scr/utils/Gnode.java: Base saving unit structure shared with normal Naive Bayes Classifier

To print confidence intervial result of 20 classificiation problems, see[scr/plots/ConfidenceInterval.java]. The result will show on consoles as source code of latex. You need copy it on Latex to see the table.

To print Data vs Accuracy linear plot of text classification problem, see[scr/plots/DALinearPlot.java].The result will show on consoles as source code of latex. You need copy it on Latex to see the table.

To print Feature vs Accuracy linear plot of text classification problem, see[scr/plots/FALinearPlot.java].The result will show on consoles as source code of latex. You need copy it on Latex to see the table.

To print time comsuming table of 20 classification problems, see[scr/plot/TimeConsumingTable]. The result will show on consoles as source code of latex. You need copy it on Latex to see the table.
