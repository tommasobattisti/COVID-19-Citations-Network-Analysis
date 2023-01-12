# Coronaviruses
## Identifying key papers within a wide citational landscape

![Modularity view of papers' network](https://raw.githubusercontent.com/Postitisnt/COVID-19-Citations-Network-Analysis/main/Data/imgs/Papers_Modularity.png)

### Context
------------------
The ongoing COVID-19 pandemic has had a profound effect on the scientific community, with researchers from various disciplines collaborating to gain insights into the virus and develop effective treatments and vaccines. 

In general, Coronaviruses have been significant research topics for many years, and the current pandemic has highlighted the importance of cross-disciplinary collaboration in addressing global challenges. For many years, experts from various disciplines have investigated the characteristics and the effects of these viruses, resulting in a wide and manifold array of citations in the scientific literature.

### Objective
------------------
In this study, we analyze the citational context of papers related with Coronaviruses, in order to illustrate a particular workflow we devised to analyze context-specific citational landscapes. 

To do that, we relied on the field of Bibliometrics, a sub-field of Library and Information Science that focuses on the use of quantitative methods to study the production, dissemination, and reception of research publications. 
In particular, this work uses methodologies from Graph Theory, Network Analysis, and Citation Analysis to retrieve and investigate citational patterns.

The corpus we focused on includes articles about Coronaviruses and their citational panorama, covering a time span that goes from the year 1904 to the 2020. By mapping the connections between them, we hope to gain a better understanding of the evolution of Coronaviruses’ research over a long period of time, including the recent emergence of COVID-19.

### Structure
------------------
This repository contains:
- DATA -> in which you can find all the data used to build and analyze the networks;
- GEPHI_FILES -> in which are stored some `.gephi` representations of the networks investigated;
- NOTEBOOKS -> in which are contained all the Jupyter Notebooks that details the three core parts of the study, as well as a different definition of the <i>NetworkX</i> `lattice_reference` function. <b>Additionally, you can find a report of the project in the `.pdf` format</b>;
- GML_NETWORKS -> in which are stored the `.gml` format networks on which we developed the study.

