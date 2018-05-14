# Strassens Matrix Multiplication
Parallel implementation of Strassen's Matrix Multiplication algorithm on small clusters (Tested on a cluster of size 52) using MPI written mainly in C. <br/> 
Speed up ranging from 2 % to 18 % is observed on large square matrices. <br/>

Please see the report for further details, analysis and noval contributions. <br/>

# Instructions:
Compilation: mpicc -o exec_file strassen.c -lm <br/>
Execution: mpiexec -n <no.of.processors> exec_file <br/>

# Sample output:

![Screenshot](https://github.com/karthikVenkataramana/Strassens/blob/master/Screeshot%20for%20n%20%3D1024%20and%20n%3D4.PNG) <br/><br/>

![Screenshot](https://github.com/karthikVenkataramana/Strassens/blob/master/Screeshot%20for%20n%20%3D1024.PNG) <br/>
