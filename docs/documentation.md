## XFoil

Uma aplicação que parece interessante para avaliação de hélices é o XFoil. Xfoil is a 2D flow solver that predicts airfoil performance (such as coefficeint of lift and drag) remarkably well (See for instance https://doi.org/10.2514/6.2008-7345)

## Exemplo surrogate model para previsão de hélices de duas pás

https://github.com/hnrosa/uiuc-propeller

## Exemplos previsão com base em Arvore de Decisão

https://github.com/Raghu-45/Prediction-of-UIUC-Propeller-efficiency
https://github.com/sujandsilva/Prediction-of-UIUC-Propeller-Efficiency-using-Decision-Tree-Regression-

## Dataset UIUC tratado

https://www.kaggle.com/datasets/heitornunes/uiuc-propeller-database
https://www.kaggle.com/datasets/williecosta/compiled-pdf-paper-much-easier-to-read

## UIUC Data

UIUC Propeller Database
Researchers like Brandt, Selig, Ananda and Deters conducted tests on various small propellers, to counteract this lack of documentation on propellers at LRN. Brandt [1] conducted tests on 79 propellers, where almost all of these propellers had a diameter ranging from 9 to 11 inches. These propellers were from different brands: Aeronaut, APC, Graupner, GWS, Kavon, Kyosho, Master Airscrew, Rev up and Zingali. Deters [2] conducted another study, where two types of propellers were tested, namely from the brands: APC, Crazyflie, E-Flite, GWS, KP, Micro Invent, Plantraco, Union and Vapor and ones that have been have been 3D-printed, named: DA4002, DA4022, DA4052 and NR640.

These tests were performed in the UIUC subsonic wind tunnel. The wind tunnel is an open-return type with a 7.5:1 contraction ration. The rectangular test section is nominally 0.853 x 1.219 m in cross section and 2.438 m long. Over the length of the test section, the width increases by approximately 0.0127 m to account for boundary-layer growth along the tunnel sidewalls. Test section speeds are variable up to 71.53 m/s via a 93.25 kW AC motor connected to a five-bladed fan. For the tests presented in reference [2], the maximum tunnel speed used was 24.38 m/s. To ensure good flow quality in the test section, the wind-tunnel settling chamber contains a 0.1016 m thick honeycomb in addition to four anti-turbulence screens.

Referência do Texto: https://ubibliorum.ubi.pt/handle/10400.6/8437

[1] M. S. Selig and J. B. Brandt, "Propeller Performance Data at Low Reynolds Numbers," in 49th AIAA Aerospace Sciences Meeting, Orlando, FL, 2011.

[2] R. W. Deters, G. K. Ananda and M. S. Selig, "Reynolds Number Effects on the Performance of Small-Scale Propellers," in AIAA Aviation, Atlanta, GA, 2014.

## UIUC Data para o meu trabalho

Usando os dados dos volumes 1-3 do dataset UIUC, Rosa gerou 3 arquivos CSV contemplando os dados dos experimentos (UIUC) e adicionando algumas outras informações complementares, estes arquivos tem a seguinte estrutura:

* **PropName**: Propeller's Name.
* **BladeName**: Blade's Name.
* **Family**: Propeller's Brand.
* **B**: Number of Blades.
* **D**: Propeller's Diameter.
* **P**: Propeller's Pitch.
* **J**: Advanced Ratio Input.
* **N**: RPM Rotation Input.
* **CT**: Thrust Coefficient Output.
* **CP**: Power Coefficient Output.
* **eta**: Efficiency Output.

https://www.kaggle.com/datasets/heitornunes/uiuc-propeller-database

Rosa, Heitor Nunes Performance prediction of small propellers with XGBoost: effects of the solidity imputation process by regression methods. Universidade Estadual Paulista (Unesp), 2022. Avaliable in: http://hdl.handle.net/11449/217119.

## Surrogate Models

Surrogate models, or metamodels, are compact scalable analytic models that estimate the results of complex tests, based on a limited set of data obtained from experimentation. These are also called response surface models (RSM), emulators, auxiliary models, repro-models, etc.

Surrogate models are a cheaper and easier solution for a test or simulation that is expensive or complex to complete because most design problems require complex experiments or simulations to evaluate certain parameters. The main goal of surrogate modeling is to achieve optimal design while reducing the number of design iterations, lowering the costs and improving overall quality. This is possible by going through a process known as curve fitting or function approximation.

Referência do Texto: https://ubibliorum.ubi.pt/handle/10400.6/8437

## JBLADE

JBLADE is an open-source propeller design and analysis code developed in UBI, by Morgado and Silvestre [1], as part of a PhD thesis at UBI. It uses a modified BEM theory to account for the 3D flow equilibrium. It can estimate the performance curves of a given propeller, after the analysis it shows the results in a graphical interface to make it easier to build and analyze the simulations.

Referência do Texto: https://ubibliorum.ubi.pt/handle/10400.6/8437

[1] J. Morgado, JBLADE v17 Tutorial, Covilhã, 2013

## QPROP Propeller/Windmill analysis and design

QPROP is an analysis program, created by professor Mark J. Drela [1], from Massachusetts Institute of Technology (MIT), which is based on a theoretical aerodynamic formulation that uses an extension of the classical blade-element/vortex formulation, as explained in the document [2], shows as output the analysis of the performance of a propeller-motor combination.

The formulation is based on an extended version of the blade-element/vortex method. This extension, implemented by Larrabee [3], is in the correct accounting of the propeller’s self-induction, making QPROP accurate for very high disk loadings [4].

Referência do Texto: https://ubibliorum.ubi.pt/handle/10400.6/8437

[1] M. Drela, "QPROP Propeller/Windmill Analysis and Design," 23 12 3007. [Online]. Available: http://web.mit.edu/drela/Public/web/qprop/. [Accessed 5 September 2018].

[2] M. Drela, "QPROP Formulation," MIT, 2006.

[3] E. E. Larrabee and S. E. French, "Minimum induced loss windmills and propellers," Journal of Wind Engineering and Industrial Aerodynamics, pp. 15:317-327, 1983.

[4] M. Drela, "QPROP User Guide," 6 July 2007. [Online]. Available: http://web.mit.edu/drela/Public/web/qprop/qprop_doc.txt. [Accessed 28 September 2018].

