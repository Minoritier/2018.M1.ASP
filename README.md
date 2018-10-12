# Applied Stochastic Processes (FIN 514, 2018-19 Module 1)

## Announcements
* HW3 ([Starter Code](py/HW3/TestCode_BasketSpread.ipynb)) is due by 10.9 (Tues)
* HW2 ([Starter Code](py/HW2/HW2.ipynb)) is due by 9.21 (Fri), [Solution](py/HW2/HW2_Solution.ipynb) was uploaded.
* Python Crash Course will be on 9.12 (Wed) **1:30 PM** (changed). Class mailing list created.
* Email is the preferred method of communication. Class mailing list will be created as PHBS.ASP@allmail.net.

## Course Slides and Other Resources
* Prelims: [Probability Statistics Review](files/Prob_Stat_Review.pdf) | [MC Method](files/MCmethod.pdf) ([Py demo](py/MC_Demo.ipynb))
* Past Exam: [2017-18](files/ASP2017_Midterm.pdf)
* Black-Scholes model ([Py demo](py/BlackScholes_ImpliedVol.ipynb), [MC demo](py/BlackScholes_MC.ipynb)): Also see Ch. 10 of [StoFin Course Notes](https://github.com/PHBS/2017.M3.StoFin/blob/master/files/SCFA_Notes.pdf)
* Normal (Bachelier) model ([Slides](files/Normal_Model.pdf))
* Implied volatility ([Slides](files/ImpVol.pdf), [Py demo](py/BlackScholes_ImpliedVol.ipynb))
* Spread/Basket options ([Slides](files/SpreadBasketOption.pdf))
* SABR model ([Slides](files/SABRmodel.pdf))
* Copula ([Slides](files/Copula.pdf), [Py demo](py/Demo_Copula.ipynb))
* ...


## Lectures
* __18__ (11.09 Tues): Course project presentation
* ...
* __13__ (10.23 Tues): Midterm exam
* ....
* __NO CLASS__ on 10.16 Tues
* ...
* __11__ (10.12 Fri): SABR model ([Slides](files/SABRmodel.pdf)): Conditional MC method
* __10__ (10.09 Tues): HW2 review, SABR model ([Slides](files/SABRmodel.pdf)), Stochastic Finance review
* __09__ (09.28 Fri): SABR model([Slides](files/SABRmodel.pdf): Volatility smile, Local volatility model)
* __08__ (09.25 Tues): [Spread/Basket option implementation](py/TestCode_BasketSpread.ipynb), Debugging in Python, Import([Py Demo](py/HW4/Demo_Advanced_Import.ipynb))
* __07__ (09.21 Fri): [Black-Scholes Implementation](https://github.com/PHBS-2017-ASP-Classroom/BSMmodel_Base), Spread/Basket options ([Slides](files/SpreadBasketOption.pdf))
* __06__ (09.18 Tues): Black-Scholes and Normal models in MC ([Py Demo](py/BlackScholes_MC.ipynb)), Normal model ([Slides](files/Normal_Model.pdf)), Correlated Normal RNs ([Py Demo](py/CorrelatedNormals_Demo.ipynb))
* __05__ (09.14 Fri): HW2, Black-Scholes implementation ([Py Demo](py/BlackScholes_FunctionVsClass.ipynb)), Implied volatility ([Slides](files/ImpVol.pdf), [Py demo](py/BlackScholes_ImpliedVol.ipynb))
* __04__ (__09.12 Wed__ instead of __10.16 Tues__): Python crash course ([Basic](py/PythonCrashCourse_Derek_Banas.ipynb) | [Numpy](py/PythonCrashCourse_Numpy.ipynb)).  More [cheatsheets](https://ehmatthes.github.io/pcc/cheatsheets/README.html) also available in [MLF CMS](http://cms.phbs.pku.edu.cn/claroline/document/document.php?cidReset=true&cidReq=FN570).
* __03__ (09.11 Tues): Continued ([Py demo](py/MC_Demo.ipynb))
* __02__ (09.07 Fri): Scientific computing, Monte Carlo method, Random number generation ([Slides](files/MCmethod.pdf)). 
* __01__ (09.04 Tues): Course overview ([Syllabus](files/syllabus.pdf)), Probability Statistics Review ([Slides](files/Prob_Stat_Review.pdf))

## Homeworks:
* ### __Set 3__ [Due by 10.9 Tues] Pricing basket and spread option using MC [Starter Code](py/HW3/TestCode_BasketSpread.ipynb)
* ### __Set 2__ [Due by 9.21 Fri] Pricing corporate (default) bond by MC simulation. [Starter Code](py/HW2/HW2.ipynb), [Solution](py/HW2/HW2_Solution.ipynb)

* ### __Set 1__ [Due by 09.11 Tues]
  * Register on Github.com and send your ID to TA via email. Use your __full name__ in your profile. Accept invitation to the [PHBS organization](https://github.com/orgs/PHBS/people) from TA. Install [Github Desktop](https://desktop.github.com/) (available on `Machine Learning` [CMS](http://cms.phbs.pku.edu.cn/claroline/course/index.php?cid=FN570)). 
  * Install [Anaconda](https://www.anaconda.com/download/) Python distribution (__3.X version__, 2.X version). Anaconda distribution is core Python + useful scientific computation libraries (e.g., numpy, scipy, pandas) + package management system (pip or conda)
  * Send the screenshot of both softwares installed to TA. (Example: [Github Desktop](files/Choi_Jaehyuk_Github.png), [Anaconda Spyder](files/Choi_Jaehyuk_Python.png))

## Course Project:

You are very welcome to do the project on your own original idea and you will get additional credit for that. Otherwise, pick one from my suggestions which are basically understanding and implementing literatures. The github repository for the project should be consist of

* Core implementation (.py): python class and functions
  * Make sure to comment in detail.
  * Put them in a separate subfolder (e.g., option_models) Do not mix with testing/manual notebook files
* Documentation/Manual (.ipynb): one Jupyter notebook file briefly describing the method (base theory, equations, SDE, strength/weakness, etc), the function prototype and arguments (manual style) and the usage examples
  * The best examples are from numpy documentation: [example](https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.polynomials.hermite.html)
* Validation/Test (.ipynb): one Jupyter notebook file briefly test the code/model.
  * Be a bit creative here
  * BSM/Normal model: make sure to include the analytic-vs-numerical risk test.
  * SV (SABR/Heston): make sure that the price converge to BSM/Normal if alpha(vov parameter) goes to 0
  * Spread/Basket: make sure that the price is same as single asset BSM if the weigit is 1 for only one asset and zero otherwise.
  
Other guidelines for the course project:
* The contribution will be individually graded. Make sure to show the contribution via github desktop commits (not online upload).
* The presentation next week doesn't have to be complete. Show your plan and understanding so far, e.g. function prototypes & arguments, etc and the tests to put on.

## Classes: 
* Lectures: Tues & Fri 1:30 – 3:20 PM
* Venue: PHBS Building, Room 211

## Instructor: [Jaehyuk Choi](http://www.jaehyukchoi.net/phbs_en)
* Office: PHBS Building, Room 755
* Phone: 86-755-2603-0568
* Email: jaehyuk@phbs.pku.edu.cn
* Office Hour: Tues & Fri 10:30 – 11:30 AM or by appointment

## Teaching Assistance: Yeda Du (都业达)
* Email: 1601213511@sz.pku.edu.cn
* TA Office Hour: Mon and Thurs, 10:30 AM - 12:30 PM (Room 213/214)

## Course overview: 
Applied Stochastic Processes (ASP) is intended for the students who are
seeking advanced knowledge in stochastic calculus and are eventually interested in the jobs in
financial engineering. As the name indicates, the course will emphasis on applications such as
numerical calculation and programming. On completion of this course, the students will learn
how financial observations (e.g. stock prices and FX rate) are modelled with stochastic
processes and how they can be computed using analytics or computer simulations.

## Prerequisites: 
[Stochastic Finance](https://github.com/PHBS/2017.M3.StoFin) (FIN 519), a year 1 required course for quantitative finance program, is also highly recommended as it provides theoretical background. Undergraduate-level knowledge in probability, statistics, linear algebra and programming skill (Python) are highly recommended. The students without these recommended prerequisites are expected to take extra efforts.

##  Textbooks and Reading Materials
* Monte Carlo Methods in Finance by Peter Jaeckel
* Option Valuation Under Stochastic Volatility by Alan Lewis
* [Stochastic Calculus and Financial Applications](http://www-stat.wharton.upenn.edu/~steele/StochasticCalculus.html) by J. Michael Steele
([Stochastic finance course notes](https://github.com/PHBS/2016.M3.StoFin/blob/master/files/Notes%20Steele.pdf))

## Assessment/Grading Details
Attendance 20%, Mid-term Exam 30%, Assignments 20%, Course Project 30%
* __Midterm exam__: 10.23 Tues. Open-book exam without computer/phone/calculator use. No final exam.
* __Course project__: Presentation (11.09 Fri). Group up to 2 people.
* __Attendance__: Randomly checked. The score is calculated as 20 – 2`x`(#of absence). Leave request should be made 24 hours before with supporting documents, except for emergency. Job interview/internship cannot be a valid reason for leave
* __Grade__ in letters (e.g., A+, A-, ... ,D+, D, F). __A- or above < 30% and C+ or below > 10%__.
