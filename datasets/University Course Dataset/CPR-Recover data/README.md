# CPR-Recover data

Release 1, 2017-01-10

This is data used in this publication:
> Recovering Concept Prerequisite Relations from University Course Dependencies. 
> Chen Liang, Jianbo Ye, Zhaohui Wu, Bart Pursel, and C. Lee Giles.
> In the 7th Symposium on Educational Advances in Artificial Intelligence, 2017.

The files are:

- *cs_courses.csv*: These are CS-related course information collected from 11 U.S
universities (Carnegie Mellon University, Stanford University, the Massachusetts Institute
of Technology, Princeton University, the California Institute of Technology, Purdue University,
University of Maryland, Michigan State University, Pennsylvania State University, University of
Illinois, and University of Iowa). Each line is formatted as "\<Course_id\>,\<Course_description\>". Note the course titles are located at the begining of the description.

- *cs_edges.csv*: There are course prerequisite information. Each line "\<course_1\>,\<course_2\>" represents \<course_2\> is a prerequisite for \<course_1\>.

- *cs_annotations.tsv*: These are annotation results for candidate pairs generated from above CS courses. Please refer to the "Data Labeling" section for more details. Each line is formatted as "\<Concept_A\>,\<Concept_B\>,\<Annotator_1\>...\<Annotator_13\>". Each pair gets labels from three different annotators. Valid labels are:
  1 B is a prerequisite of A.
  2 A is a prerequisite of B.
  3 There is no prerequisite relation between A and B.
  
- *cs_preqs.csv*: These are concept prerequisite pairs exported from the above annotation by using majority vote. Each line "\<Concept_A\>,\<Concept_B\>" represents that B is a prerequisite of A.

<strong>Note</strong>: As described in the paper, Wikipedia concepts in this data are all extracted with the help of [Wikipedia-miner](https://github.com/dnmilne/wikipediaminer). You can also try other Wikification/Entity linking methods to extract Wiki concepts from course descriptions. In that case, even though our labeled prerequisite pairs perhaps will not cover all candidate pairs, we believe this annotation still covers most of them and can save you lots of time when collecting prerequisite labels.

If you have any problems, please contact Chen Liang at <cul226@ist.psu.edu>.

## License
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/). 

![Alt](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)
