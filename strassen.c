/*
      Author: Karthik Venkataramana Pemmaraju
      EUID : kp0327
      CSCE 5160 Term Project
      Date : 02/10/2016 - 12/10/2016
      Title: Parallel implementation of Strassen's matrix multiplication algorithm.

      Restriction:  
      1) m=k=n (No.of rows = No. of columns)
      2) The no. of processors are powers of 2 exluding 2 i.e 4,8,16,32 and so on...
      
      Compilation: mpicc -o exec_file strassen.c -lm
            Execution: mpiexec -n <no.of.processors> exec_file
      
      Outputs are attached in outputs.txt file.

*/
  #include <stdio.h>
  #include <mpi.h> 
  #include <stdlib.h>
  #include <math.h>

    // Global definitions to access the matrices
    #define A(i, j) A[(i) * (n) + (j)]
    #define B(i, j) B[(i) * (n) + (j)]
    #define C(i, j) C[(i) * (n) + (j)]

      // For colors.
   #define ANSI_COLOR_RED "\x1b[31m"
   #define ANSI_COLOR_RESET "\x1b[0m"
   #define ANSI_COLOR_YELLOW "\x1b[33m"
   #define ANSI_COLOR_BLUE "\x1b[34m"
   #define ANSI_COLOR_CYAN "\x1b[36m"

     // Size of the matrix
  const int n = 1024;
    
     //Function definitions. Explained later.
  double * Serial_Multiply(double * , double * , double * , int);
  void print_matrix(double * , int, int);
  double * Parallel_Multiply(double * , double * , double * , int, int, int, int, int);
  double * Form_Matrix(double * , double * , double * , double * , double * , int);
  double findAverage(double[], int);
  
  // Everything starts from here.
  int main(int argc, char * * argv) {
    int my_rank, p;
    double * A, * B;
    double * C;
    int i, j;
    int rows, columns;
    MPI_Request req[8];
    MPI_Status status;
    MPI_Datatype mysubarray, subarrtype; // used for sending 2-D blocks to processors.
    double initialization_time, communication_time, finalize_time, computation_time;
    double start, end;
    double temp_start, temp_end;
    double serial_average_time, strassens_time, parallel_multiply_time, serial_time[3], parallel_time[3];

    MPI_Init( & argc, & argv); // Initialize the MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, & my_rank); // returns the rank of individual processors
    MPI_Comm_size(MPI_COMM_WORLD, & p); //returns the number of processors.

    start = MPI_Wtime(); // Calculating initialization time.
    A = (double * ) malloc(n * n * sizeof(double)); //n of A matrix.
    B = (double * ) malloc(n * n * sizeof(double)); //n of B matrix.
    C = (double * ) malloc(n * n * sizeof(double)); //n of C matrix.

    /******Set up our matrices**********/
    if (my_rank == 0) { //master process.
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          A(i, j) = i + j; //As used in Cannon's simply for testing purposes
          B(i, j) = i - j; // As used in Cannon's
          C(i, j) = 0.0; //Initialize C with 0's
        } //end for of columns
      } // end for of rows
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    initialization_time = end - start;  //Time taken to initialize matrices.
    if (my_rank == 0) {
      printf("/**********************************Original B Matrix****************************/\n");
      print_matrix(B, n, n);
      j = 0;
      while (j < 3) {  //Calculate 3 times and find average
        start = MPI_Wtime();
        // print_matrix(C,n,n);
        C = Serial_Multiply(A, B, C, n);
        end = MPI_Wtime();
        serial_time[j] = initialization_time + (end - start);
        j++;
      }
      serial_average_time = findAverage(serial_time, 3); // Calcualting serial average execution time.
      printf("/*********************************Serial multiplication***********************************************/ \n");
      print_matrix(C, n, n); //prints the product matrix in C
      printf("Serial execution time is %f seconds\n", serial_average_time);
    } //Our Master processor ends.

    MPI_Barrier(MPI_COMM_WORLD);
    j = 0;
    while (j < 3) {  //Standard Parallel matrix multiply repeated 3 times
      start = MPI_Wtime();
      C = Parallel_Multiply(A, B, C, 0, my_rank, p, n, n); // Standard Multiply, result stored in C.
      end = MPI_Wtime();
      parallel_time[j] = initialization_time + (end - start); 
      j++;
    }
    parallel_multiply_time = findAverage(parallel_time, 3); // Found Average time.
    if (my_rank == 0) {
      printf("/***********************Conventional Parallel Matrix multiplication********************/\n ");
      print_matrix(C, n, n); // printing C.
    }
      
       MPI_Barrier(MPI_COMM_WORLD); //wait till every processor has called.

    start = MPI_Wtime(); // Start clock for Strassens 
    int starts[2] = {
      0,
      0
    }; //Starting from 0,0
    int bigns[2] = {
      n,
      n
    }; // Our total n 
    int subns[2] = {
      n / 2, //rows
      n / 2 //columns
    };  
    MPI_Type_create_subarray(2, bigns, subns, starts, MPI_ORDER_C, MPI_DOUBLE, & mysubarray); // Creates a sub array of 2d and stores in MPI data type mysubarray
    MPI_Type_create_resized(mysubarray, 0, (n / 2) * (n / 2) * sizeof(double), & subarrtype); // Each subarray has a total of n/sqrt(p) elements.
    MPI_Type_commit( & subarrtype); // You want to commit the main data type you wanna use later

    rows = (n / 2); //Since A11,A12,A21,A22 have sizes of (n/2) * (n/2)
    columns = (n / 2);
    int no_of_elements = rows * columns;
    double * local_A = (double * ) malloc(no_of_elements * sizeof(double)); // local A matrix for the processors
    double * local_B = (double * ) malloc(no_of_elements * sizeof(double)); // local B matrix for the processors
    double * local_C = (double * ) malloc(no_of_elements * sizeof(double)); // local C matrix for the processors

      // Sending A12 to Processor 1, A21 to Processor 2, A22 to processor 3 
    if (my_rank == 0) {
      /* We keep A11 with processor 0 and send A12,A21,A22 to processors 1,2,3 respectively.*/
      MPI_Isend(A, 1, subarrtype, 0, 0, MPI_COMM_WORLD, & req[0]); //Sending A11 to itself to change data type from subarrtype to MPI_Double
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, & req[0]); //Receiving A11
      MPI_Wait( & req[0], & status);
      MPI_Isend(B, 1, subarrtype, 0, 0, MPI_COMM_WORLD, & req[0]); //Sending B11
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, & req[0]); //Receiving B11
      MPI_Wait( & req[0], & status);
      MPI_Isend(C, 1, subarrtype, 0, 0, MPI_COMM_WORLD, & req[0]); //Sending C11
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, & req[0]); //Receiving C11
      MPI_Wait( & req[0], & status);
      MPI_Isend(A + n / 2, 1, subarrtype, 1, 1, MPI_COMM_WORLD, & req[1]); //Sending A12, A12 starting address is A+n/2 
      MPI_Isend(B + n / 2, 1, subarrtype, 1, 1, MPI_COMM_WORLD, & req[1]); //Sending B12
      MPI_Isend(C + n / 2, 1, subarrtype, 1, 1, MPI_COMM_WORLD, & req[1]); //Sending C12

      MPI_Isend(A + ((n / 2) * (n)), 1, subarrtype, 2, 2, MPI_COMM_WORLD, & req[2]); //Sending A21, starting address is A + ((n / 2) * (n))
      MPI_Isend(B + ((n / 2) * (n)), 1, subarrtype, 2, 2, MPI_COMM_WORLD, & req[2]); //Sending B21
      MPI_Isend(C + ((n / 2) * (n)), 1, subarrtype, 2, 2, MPI_COMM_WORLD, & req[2]); //Sending C21

      MPI_Isend(A + ((n / 2) * n + (n - 1)), 1, subarrtype, 3, 3, MPI_COMM_WORLD, & req[3]); //Sending A22 , starting address is A + ((n / 2) * (n)) + n/2
      MPI_Isend(B + ((n / 2) * n + (n - 1)), 1, subarrtype, 3, 3, MPI_COMM_WORLD, & req[3]); //Sending B22 
      MPI_Isend(C + ((n / 2) * n + (n - 1)), 1, subarrtype, 3, 3, MPI_COMM_WORLD, & req[3]); //Sending C22 

    }

    if (my_rank == 1) {
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, & req[1]); // Take A12
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, & req[1]); // Take B12
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, & req[1]); // Take c12
      MPI_Wait( & req[1], & status);
    }
    if (my_rank == 2) {
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, & req[2]); //Take A21
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, & req[2]); //Take B21
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, & req[2]); //Take C21

      MPI_Wait( & req[2], & status);
    }
    if (my_rank == 3) {
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, & req[3]); //Take A22 
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, & req[3]); //Take B22 
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, & req[3]); //Take C22 
      MPI_Wait( & req[3], & status);
    }
    // MPI_Scatterv(A, sendcounts, displs, subarrtype, local_A,
    // no_of_elements, MPI_DOUBLE,
    // 0, MPI_COMM_WORLD); 
    MPI_Type_free( & subarrtype); //free the MPI data type

  /* We completed sending the matrices to different processors */
    MPI_Barrier(MPI_COMM_WORLD);

    double * T1 = (double * ) malloc(sizeof(double) * no_of_elements); //rows = n/2 , columns =n/2
    double * T2 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T3 = (double * ) malloc(sizeof(double) * no_of_elements); //rows = n/2 , columns =n/2
    double * T4 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T5 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T6 = (double * ) malloc(sizeof(double) * no_of_elements); //rows = n/2 , columns =n/2
    double * T7 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T8 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T9 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T10 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T11 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T12 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T13 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * T14 = (double * ) malloc(sizeof(double) * no_of_elements);
    double * P1 = (double * ) malloc(sizeof(double) * (no_of_elements));
    double * P2 = (double * ) malloc(sizeof(double) * (no_of_elements));
    double * P3 = (double * ) malloc(sizeof(double) * (no_of_elements));
    double * P4 = (double * ) malloc(sizeof(double) * (no_of_elements));
    double * P5 = (double * ) malloc(sizeof(double) * (no_of_elements));
    double * P6 = (double * ) malloc(sizeof(double) * (no_of_elements));
    double * P7 = (double * ) malloc(sizeof(double) * (no_of_elements));

    if (T1 == NULL || T2 == NULL || T3 == NULL || T4 == NULL || T5 == NULL || T6 == NULL || T7 == NULL || T8 == NULL || T9 == NULL || T10 == NULL || P1 == NULL || P2 == NULL || P3 == NULL || P4 == NULL)
      printf("Error allocating memory\n");
    // Processor 0

    if (my_rank == 0) {
      // Our processor 0 calculates P1 and P7 
        /* 
      We calculate P1 = (A11+A22) (B11+B222)
      P7 = (A12-A22) (B21+B22)
      on processor O.
    */
      double * local_A22 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * local_B22 = (double * ) malloc(sizeof(double) * no_of_elements);

      double * local_A12 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * local_B21 = (double * ) malloc(sizeof(double) * no_of_elements);

      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, & req[0]); //send A11 to processor 1
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, & req[4]); //send A11 to processor 3

      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 2, 400, MPI_COMM_WORLD, & req[3]); //send B11 to processor 3

      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 3, 600, MPI_COMM_WORLD, & req[4]); //send B11 to processor 3

      MPI_Irecv(local_A22, no_of_elements, MPI_DOUBLE, 3, 100, MPI_COMM_WORLD, & req[0]); // Take A22 from processor 3
      MPI_Irecv(local_B22, no_of_elements, MPI_DOUBLE, 3, 200, MPI_COMM_WORLD, & req[0]); // Take B22 from processor 3
      MPI_Wait( & req[0], & status);

      MPI_Irecv(local_A12, no_of_elements, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, & req[1]); // Take A12 from 2nd processor
      MPI_Wait( & req[1], & status);

      MPI_Irecv(local_B21, no_of_elements, MPI_DOUBLE, 2, 200, MPI_COMM_WORLD, & req[2]); // Take B21 from processor 3
      MPI_Wait( & req[0], & status);

      for (i = 0; i < no_of_elements; i++) {
        * (T1 + i) = * (local_A + i) + * (local_A22 + i); // T1= A11 +A22 
        * (T2 + i) = * (local_B + i) + * (local_B22 + i); // T2 = B11+ B22
        * (T3 + i) = * (local_A12 + i) - * (local_A22 + i); // T3= A12 - A22 
        * (T4 + i) = * (local_B21 + i) + * (local_B22 + i); // T4 = B21+ B22
      }
       //Set to NULL and free the pointers.
      local_A22 = NULL;
      local_B22 = NULL;
      local_A12 = NULL;
      local_B21 = NULL;
      free(local_A22);
      free(local_B22);
      free(local_A12);
      free(local_B21);
    }

    // Processor 1
    if (my_rank == 1) {
      //Processor 1 calculates P3 and P5 for that I need A11 and B22 
      double * local_A11 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * local_B22 = (double * ) malloc(sizeof(double) * no_of_elements);

      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, & req[1]); //send A12 to processor 0

      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, & req[1]); //send B12 to processor 3

      MPI_Irecv(local_A11, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, & req[0]); // Take A11 from processor 0
      MPI_Irecv(local_B22, no_of_elements, MPI_DOUBLE, 3, 200, MPI_COMM_WORLD, & req[0]); // Take B22 from processor 3
      MPI_Wait( & req[0], & status);

      for (i = 0; i < no_of_elements; i++) { 
        * (T5 + i) = * (local_B + i) - * (local_B22 + i); // T5 = B11 - B22 
        * (T6 + i) = * (local_A11 + i) + * (local_A + i); //T6 =  A11+A12
        * (T7 + i) = * (local_B22 + i); //T7 = B22
        * (T14 + i) = * (local_A11 + i); // T14 =A11
      }
      free(local_A11);
      free(local_B22);

    }
    /*** Processor 1 done computing P3 and P5 ***/

    if (my_rank == 2) {
      // Processor 2 computes P2 and P4 for that I need A22 and B11.
      double * local_A22 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * local_B11 = (double * ) malloc(sizeof(double) * no_of_elements);

      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, & req[2]); // send B21 to processor 0
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, & req[2]); // send A21 to processor 3
      // For P2, I need A22 and B11. Let's get them.
      MPI_Irecv(local_A22, no_of_elements, MPI_DOUBLE, 3, 100, MPI_COMM_WORLD, & req[3]); // Take A22 from processor 3
      MPI_Wait( & req[3], & status);
      MPI_Irecv(local_B11, no_of_elements, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD, & req[3]); // Take B11 from processor 0
      MPI_Wait( & req[3], & status);

      for (i = 0; i < no_of_elements; i++) { 
        * (T8 + i) = * (local_A + i) + * (local_A22 + i); // T8 = A21 + A22 
        * (T9 + i) = * (local_B11 + i); // T9 = B11 
        * (T10 + i) = * (local_A22 + i); //T10 = A22
        * (T11 + i) = * (local_B + i) - * (local_B11 + i); //T11= B21-B11
      }

      free(local_A22);
      free(local_B11);
    }
    // Processor 3.
    if (my_rank == 3) {
   
      double * local_A21 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * local_A11 = (double * ) malloc(sizeof(double) * no_of_elements);
      double * local_B12 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * local_B11 = (double * ) malloc(sizeof(double) * no_of_elements);
      //Processor 3 calculates P6 for that I need A11,A21,B11,B12.
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, & req[0]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, & req[0]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 1, 200, MPI_COMM_WORLD, & req[0]); //send B22 to processor 0
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 2, 100, MPI_COMM_WORLD, & req[3]); //send A22 to processor 2

      MPI_Irecv(local_A21, no_of_elements, MPI_DOUBLE, 2, 500, MPI_COMM_WORLD, & req[4]); // Take A21 from processor 2

      MPI_Irecv(local_B11, no_of_elements, MPI_DOUBLE, 0, 600, MPI_COMM_WORLD, & req[4]); // Take B11 from processor 0

      MPI_Irecv(local_B12, no_of_elements, MPI_DOUBLE, 1, 500, MPI_COMM_WORLD, & req[4]); // Take B12 from processor 1

      MPI_Irecv(local_A11, no_of_elements, MPI_DOUBLE, 0, 500, MPI_COMM_WORLD, & req[5]); // Take A11 from processor 0
      MPI_Wait( & req[4], & status);
      MPI_Wait( & req[5], & status);

      for (i = 0; i < no_of_elements; i++) {
        * (T12 + i) = * (local_A21 + i) - * (local_A11 + i); // T12 = A21 - A11 
        * (T13 + i) = * (local_B11 + i) + * (local_B12 + i); // T13 = B11 + B12 
      }

      free(local_B12);
      free(local_A21);
      free(local_A11);
      free(local_B11);

    }
      /* Calculate P1..7 in parallel */
    P1 = Parallel_Multiply(T1, T2, P1, 0, my_rank, p, rows, columns); // P1= T1*T2
    P7 = Parallel_Multiply(T3, T4, P7, 0, my_rank, p, rows, columns); // P7 =T3*T4
    P3 = Parallel_Multiply(T14, T5, P3, 1, my_rank, p, rows, columns); //P3 = T14*T5
    P5 = Parallel_Multiply(T6, T7, P5, 1, my_rank, p, rows, columns);  // P5 = T6*T7 
    P2 = Parallel_Multiply(T8, T9, P2, 2, my_rank, p, rows, columns);
    P4 = Parallel_Multiply(T10, T11, P4, 2, my_rank, p, rows, columns);
    P6 = Parallel_Multiply(T12, T13, P6, 3, my_rank, p, rows, columns);

    /* We calculated P1...7. let's get them onto our root processor(0) */

    if (my_rank == 1) {
      MPI_Isend(P3, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, & req[3]); //send P3 to processor 0
      MPI_Isend(P5, no_of_elements, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, & req[3]); //send P5 to processor 0

    }
    if (my_rank == 2) {
      MPI_Isend(P2, no_of_elements, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD, & req[4]); //send P2 to processor 0
      MPI_Isend(P4, no_of_elements, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD, & req[4]); //send P4 to processor 0

    }
    if (my_rank == 3) {
      MPI_Isend(P6, no_of_elements, MPI_DOUBLE, 0, 500, MPI_COMM_WORLD, & req[4]); //send P3 to processor 0 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Everything is on our root processor , just receiving and assembling them
    
    if (my_rank == 0) {
      MPI_Irecv(P3, no_of_elements, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, & req[3]); // Take A21 from processor 2
      MPI_Irecv(P5, no_of_elements, MPI_DOUBLE, 1, 200, MPI_COMM_WORLD, & req[3]); // Take B11 from processor 0
      MPI_Irecv(P2, no_of_elements, MPI_DOUBLE, 2, 300, MPI_COMM_WORLD, & req[4]); // Take B12 from processor 1
      MPI_Irecv(P4, no_of_elements, MPI_DOUBLE, 2, 400, MPI_COMM_WORLD, & req[4]); // Take A11 from processor 0
      MPI_Irecv(P6, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, & req[5]); // Take A11 from processor 0
      MPI_Wait( & req[3], & status);
      MPI_Wait( & req[4], & status);
      MPI_Wait( & req[5], & status);
      double * C11 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * C12 = (double * ) malloc(sizeof(double) * no_of_elements);
      double * C21 = (double * ) malloc(sizeof(double) * no_of_elements); // no of rows are n/2 and columns n/2
      double * C22 = (double * ) malloc(sizeof(double) * no_of_elements);
      // Our core logic.
      for (i = 0; i < no_of_elements; i++) { 
        * (C11 + i) = * (P1 + i) + * (P4 + i) - * (P5 + i) + * (P7 + i); // C11 = P1 + P4 - P5 + P7
        * (C12 + i) = * (P3 + i) + * (P5 + i); // C12 = P3+P5;
        * (C21 + i) = * (P2 + i) + * (P4 + i); // C21 = P2+P4;
        * (C22 + i) = * (P1 + i) + * (P3 + i) - * (P2 + i) + * (P6 + i); // C22 = P1+P3-P2+P6
      }
      // We have result but is in order C11 followed by C12 followed by C21 followed by C22 pointed by C. So we should convert it.

      C = Form_Matrix(C11, C12, C21, C22, C, n); // Converts C into desired order
      end = MPI_Wtime();
      strassens_time = initialization_time + (end - start);
      printf("/************************************Strassens******************************/ \n ");
      print_matrix(C, n, n); // printing desired product.
      free(C11);
      free(C22);
      free(C12);
      free(C21);
    }
    // //*** Free the pointers ****//
    MPI_Barrier(MPI_COMM_WORLD); // Wait till everyone catches up.
    temp_start = MPI_Wtime();
    free(A);
    free(B);
    free(C);
    end = MPI_Wtime() - temp_start;
    parallel_multiply_time = parallel_multiply_time + end;  // Our parallel matrix multiply time also includes freeeing the large pointers.
    free(local_A);
    free(local_B);
    free(local_C);
    free(P1);
    free(T1);
    free(T2);
    free(T3);
    free(T4);
    free(T5);
    free(T6);
    free(T7);
    free(T8);
    free(T9);
    free(T10);
    free(T11);
    free(T12);
    free(T13);
    free(T14);
    free(P2);
    free(P3);
    free(P4);
    free(P5);
    free(P6);
    free(P7);
    MPI_Barrier(MPI_COMM_WORLD);
    double improvement_serial, improvement_parallel;
    if (my_rank == 0) {
      temp_end = MPI_Wtime();
      finalize_time = temp_end - temp_start; // Strassens time to free the pointers.
      strassens_time = strassens_time + finalize_time; // Add it to total strassens time.
      improvement_serial = ((serial_average_time - strassens_time)) / serial_average_time * 100; //Improvement over serial.
      improvement_parallel = ((parallel_multiply_time - strassens_time) / parallel_multiply_time) * 100; //Improvement over parallel.
      printf("Strassens execution time is %f seconds and Speed up is %f \n\n", strassens_time, (serial_average_time / strassens_time));
      printf("Parallel Matrix multiplication time is %f seconds and Speed up is %f\n \n", parallel_multiply_time, (serial_average_time / parallel_multiply_time));
      printf("Improvement in execution time of Strassens over Serial matrix multiplication is %0.2f % \n \n", improvement_serial);
      printf("Improvement in execution time of Strassens over parallel matrix multiplication is %0.2f %\n ", improvement_parallel);

    }
    // // Finalize the MPI environment. 
    MPI_Finalize();
    return 0;
  }

  double * Parallel_Multiply(double * A, double * B, double * C, int root, int my_rank, int no_of_processors, int rows, int columns) {
    double from, to;
    int i, j, k;
    double sum;
    double start, end;
    if (rows % no_of_processors != 0) {
      if (my_rank == root) printf("Matrix size not divisible by number of processors\n");
      exit(-1);
    }
    int no_of_elements = rows * columns;
    double * result = (double * ) malloc(sizeof(double) * no_of_elements);
    from = my_rank * rows / no_of_processors;  // Calculating from
    to = (my_rank + 1) * rows / no_of_processors; // to
   
    MPI_Bcast(B, no_of_elements, MPI_DOUBLE, root, MPI_COMM_WORLD); // Broadcast entire matrix B to all processors
    MPI_Scatter(A, no_of_elements / no_of_processors, MPI_DOUBLE, A, no_of_elements / no_of_processors, MPI_DOUBLE, root, MPI_COMM_WORLD); // Scatter A (n/p) rows to all processors
    int loop = (to - from);
    int ci = 0;
    for (i = from; i < to; i++) {
      for (j = 0; j < columns; j++) {
        sum = 0;
        for (k = 0; k < rows; k++) {
          sum += A[ci * rows + k] * B[k * rows + j]; // Calculate C = A*B
        }
        result[ci * rows + j] = sum;
      }
      ci++;
    }
    //Gather results to root.
    MPI_Gather(result, no_of_elements / no_of_processors, MPI_DOUBLE, C, no_of_elements / no_of_processors, MPI_DOUBLE, root, MPI_COMM_WORLD);
    
    free(result);
    return C;
  }

    // Standard Serial multiplication. 
  double * Serial_Multiply(double * A, double * B, double * C, int size) {
    int i, j, k;
    double sum;
    for (i = 0; i < size; i++) {
      for (j = 0; j < size; j++) {
        sum = 0;
        for (k = 0; k < size; k++) {
          sum += A[i * size + k] * B[k * size + j]; //C = A*B; 
        }
        C[i * size + j] = sum; //end inner most loop
      } // end middle loop
    } // end outer loop

    return C;
  }
  static int count = 0; // used for colors
  void print_matrix(double * N, int rows, int columns) {

    int i, j;
    //print only 4*4 matrix.
    if (rows > 4)
      rows = 4;
    if (columns > 4)
      columns = 4;
    printf("Printing only first 4 results \n");
    for (i = 0; i < rows; i++) {
      for (j = 0; j < columns; j++) {
        if (count == 0)
          printf(ANSI_COLOR_RED "%lf \t", N[((n * i) + j)]); //row major accessing with red color.
        else if (count == 1)
          printf(ANSI_COLOR_BLUE "%lf \t", N[((n * i) + j)]); //row major accessing with blue.
        else if (count == 2)
          printf(ANSI_COLOR_YELLOW "%lf \t", N[((n * i) + j)]); //row major accessing with Yellow.
        else
          printf(ANSI_COLOR_CYAN "%lf \t", N[((n * i) + j)]); //row major accessing with Cyan.

      }
      printf(ANSI_COLOR_RESET "\n");
    }
    count++;
  }

  double * Form_Matrix(double * C11, double * C12, double * C21, double * C22, double * C, int n) {

    int i, j, i2 = 0;
    int i3 = -n / 2;
    int i4 = i3;
    int j2, j3;

      // We are trying to assemble inputs to C matrix.
    for (i = 0; i < n; i++) {
      j2 = 0; //loop variable for C12
      j3 = 0; // loop variable for C22
      for (j = 0; j < n; j++) {

        if (j < (n / 2) && (i < n / 2)) // first Quadrant
          C[i * n + j] = C11[i * (n / 2) + j]; // C11  First n/2 elements

        if (j >= n / 2 && i < n / 2) {   //Second Quadrant
          C[i * n + j] = C12[i2 * (n / 2) + j2]; //C12 
          j2++;
        }

        if (j < n / 2 && i >= n / 2) // Third quadrant.
          C[i * n + j] = C21[i3 * (n / 2) + j]; // C21

        if (i >= n / 2 && j >= n / 2) { // We are in fourth quadrant.
          C[i * n + j] = C22[i4 * (n / 2) + j3]; //C22
          j3++;
        }
      }

      i2++; //for C2
      i3++; // for C3
      i4++; // For C4
    }
    return C; // combined result.
  }
  
  // Calcuates the average of pointer array. Pretty much straight forward.
  double findAverage(double a[], int n) {

    int i;
    double average, sum = 0;
    for (i = 0; i < n; i++) {
      sum += a[i];
    }
    return (sum / n);
  }