#include <stdio.h>
#include <stdlib.h> 


double* read_signal_from_file(const char* filename, int* signal_size) {
    // verifica integridade do arquivo 
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // logic to read all the data in the file 
    int count = 0;
    double temp_val;
    while (fscanf(file, "%lf", &temp_val) == 1) {
        count++;
    }

    if (count == 0) {
        fprintf(stderr, "Error: No data found in the file.\n");
        fclose(file);
        return NULL;
    }

    *signal_size = count;

    // allocate the memory in the buffer 
    double* signal_data = (double*)malloc(sizeof(double) * count);
    if (signal_data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory.\n");
        fclose(file);
        return NULL;
    }

    // read the data into the allocated array 
    rewind(file); 
    int i = 0;
    while (i < count && fscanf(file, "%lf", &signal_data[i]) == 1) {
        i++;
    }

    fclose(file);
    return signal_data;
}


void plot_signal(const double* signal_data, int signal_size, const char* plot_title) {
    // this function gonna use gnuplot 
    // "-persistent" keeps the plot window open after the C program exits.
    FILE *gnuplot_pipe = popen("gnuplot -persistent", "w");

    if (gnuplot_pipe == NULL) {
        fprintf(stderr, "Error: gnuplot not found. Please ensure it's installed and in your PATH.\n");
        return;
    }

    // send commands to gnuplot 
    fprintf(gnuplot_pipe, "set title '%s'\n", plot_title);
    fprintf(gnuplot_pipe, "set xlabel 'Sample Number'\n");
    fprintf(gnuplot_pipe, "set ylabel 'Amplitude'\n");
    // The '-' tells gnuplot to read data from the pipe (stdin)
    fprintf(gnuplot_pipe, "plot '-' with lines title 'Signal'\n");

    //  send the data points to gnuplot 
    for (int i = 0; i < signal_size; i++) {
        fprintf(gnuplot_pipe, "%f\n", signal_data[i]);
    }

    // 'e' signals the end of the data stream to gnuplot
    fprintf(gnuplot_pipe, "e\n");

    // Flush the pipe to ensure all data is sent
    fflush(gnuplot_pipe);

    // Close the pipe
    pclose(gnuplot_pipe);
}

int main(int argc, char *argv[]) {
    // Check if a filename was provided as a command-line argument
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename.txt>\n", argv[0]);
        return 1; 
    }

    const char* filename = argv[1];
    int size = 0;
    double* signal;

    printf("Reading signal from: %s\n", filename);
    signal = read_signal_from_file(filename, &size);

    if (signal != NULL && size > 0) {
        printf("Successfully read %d samples.\n", size);
        printf("Plotting signal...\n");
        
        plot_signal(signal, size, filename);
        
        // if you dont free you die 
        free(signal);
    } else {
        fprintf(stderr, "Failed to read or process the signal.\n");
        return 1; // Indicate error
    }

    return 0; // Indicate success
}

/*

sudo apt update && sudo apt install gnuplot 

gcc -o leo leo.c 

./leo lfpHFO_wav1.txt

*/