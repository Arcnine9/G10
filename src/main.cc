/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 * You should not need to modify this file.
 */

#include <chrono>
#include <string>
#include <math.h>
#include <random>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <unistd.h>
#include "utility.h"
#include "errors.h"
#include "parser.h"
#include "y.tab.h"
#include "analysis.h"
#include "codegen.h"
#include "simulationComponents.h"
#include "simulator.h"
#include "printUtils.h"

#define YYDEBUG 1

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Codegen param
extern int is_resnet;
extern int is_inception;
extern int is_senet;
extern int batch_size;
extern int input_H;
extern int input_W;
extern int num_threads;
extern bool is_individual;
extern bool is_input_pf_only;

// CPU sim param
extern double CPU_PCIe_bandwidth_GBps;
// GPU sim param
extern double GPU_PCIe_bandwidth_GBps;
extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double GPU_malloc_uspB;
extern double GPU_free_uspB; // NOT USED FOR NOW
// SSD sim param
extern double SSD_PCIe_bandwidth_GBps;
// PCIe sim param
extern double PCIe_latency_us;  // NOT USED FOR NOW
extern int PCIe_batch_size_in_page;
// Other sim param
extern bool use_prefetch;
extern std::string migration_policy_str;
extern std::string eviction_policy_str;
extern Simulator::MigPolicy migration_policy;
extern Simulator::GPUPageTable::EvcPolicy eviction_policy;
extern int prefetch_degree;
extern int num_candidate;
extern double system_latency_us; // NOT USED FOR NOW

// Other param
//   In codegen, is_UVM specifies whether to use cudaMallocManaged
//   In simulation, is_UVM specifies whether setup is ideal (i.e. all tensor in GPU mem)
bool is_UVM = true;
//   In codegen, num_iteration specifies number of iterations to profile
//   In simulation, num_iteration specifies number of iterations to run
int num_iteration = -1;
int is_transformer = -1;
int borden = 184;

// 
extern double CPU_memory_line_GB;
extern double SSD_read_latency_us;
extern double SSD_write_latency_us;
extern double SSD_latency_us; // Upper bound
extern double delta_parameter;

// Tensor configurations
extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

// 
extern std::vector<Model_Layer*> forward_layers;
extern std::vector<Model_OP*> forward_ops;
extern std::vector<CUDAKernel> kernel_list;
extern std::vector<Tensor*> tensor_list;
extern std::vector<Hidding_Interval*> interval_list;
extern std::vector<EvictionGuide_Entry> EvictionGuide_Table;
extern std::vector<long> GPU_resident_memory_estimation;
extern std::vector<double> kernel_time_table;

// output specifications
std::string nn_model_input_file;
std::string orig_kernel_time_file;
std::string input_pf_kernel_time_file;
std::string workspace_size_file;
std::string pf_kernel_time_file;
std::string stat_output_file;
std::string output_folder_name;
// simulation switches
bool is_simulation = true;
bool output_override = false;
// profiling switches
bool is_compile = true;
bool is_run = true;
int compile_max_thread_num = -1;
bool is_cudnn = false;

// random devices
std::mt19937 rand_device;
double kernel_time_std_dev = 0;
unsigned int ran_seed = 1;
double kernel_speedup = 1;

/* Function: PrintOneToken()
 * Usage: PrintOneToken(T_Double, "3.5", val, loc);
 * -----------------------------------------------
 */
static void PrintOneToken(yytokentype token, const char *text, YYSTYPE value,
                          yyltype loc)
{
  char buffer[] = {'\'', (char) token, '\'', '\0'};
  const char *name = token >= T_Sequential ? gTokenNames[token - T_Sequential] : buffer;

  printf("%-12s line %d cols %d-%d is %s ", text,
	   loc.first_line, loc.first_column, loc.last_column, name);

  switch(token) {
    case T_IntConstant:
      printf("(value = %d)\n", value.integerConstant); break;
    case T_DoubleConstant:
      printf("(value = %g)\n", value.doubleConstant); break;
    case T_BoolConstant:
      printf("(value = %s)\n", value.boolConstant ? "true" : "false"); break;
    case T_Identifier:
	if (strcmp(text, value.identifier)) {
	  printf("(truncated to %s)\n", value.identifier);
	  break;
	}
    default:
      printf("\n"); break;
  }
}

void CheckVar(double var, std::string variable_name, bool gt=true) {
    if ((gt && var < 0) || (!gt && var > 0)) {
        eprintf("Invalid or missing <%s>, current value: %f, should be %s than 0, aborting\n", 
                variable_name.c_str(), var, gt ? "greater" : "less");
        Assert(false);
    }
}

void SimulationParamSanityCheck() {
    // parameter validation (existence)
    CheckVar(PCIe_batch_size_in_page, "PCIe_batch_size_in_page");
    CheckVar(CPU_PCIe_bandwidth_GBps, "CPU_PCIe_bandwidth_GBps");
    CheckVar(GPU_PCIe_bandwidth_GBps, "GPU_PCIe_bandwidth_GBps");
    CheckVar(SSD_PCIe_bandwidth_GBps, "SSD_PCIe_bandwidth_GBps");
    CheckVar(GPU_frequency_GHz, "GPU_frequency_GHz");
    CheckVar(GPU_memory_size_GB, "GPU_memory_size_GB");
    CheckVar(GPU_malloc_uspB, "GPU_malloc_uspB");
    CheckVar(GPU_free_uspB, "GPU_free_uspB");
    CheckVar(CPU_memory_line_GB, "CPU_memory_line_GB");

    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        Assert(eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM)
        Assert(migration_policy == Simulator::MigPolicy::DEEPUM);
    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        CheckVar(prefetch_degree, "prefetch_degree");
    else
        CheckVar(prefetch_degree, "prefetch_degree", false);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::GUIDED || 
        eviction_policy == Simulator::GPUPageTable::EvcPolicy::GUIDED_LRU ||
        eviction_policy == Simulator::GPUPageTable::EvcPolicy::PERFECT_GUIDED ||
        eviction_policy == Simulator::GPUPageTable::EvcPolicy::PERFECT_GUIDED_LRU)
        CheckVar(num_candidate, "num_candidate");
    else
        CheckVar(num_candidate, "num_candidate", false);
    CheckVar(num_iteration, "num_iteration");

    // parameter validation (value)
    if (SSD_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid SSD Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (CPU_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid CPU Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (SSD_PCIe_bandwidth_GBps > CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported SSD Bandwidth [%f] > CPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (GPU_PCIe_bandwidth_GBps > SSD_PCIe_bandwidth_GBps + CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported GPU Bandwidth [%f] > SSD Bandwidth [%f] + CPU Bandwidth [%f]\n",
                GPU_PCIe_bandwidth_GBps, SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (kernel_speedup <= 0) {
        eprintf("Invalid kernel speedup [%f]\n", kernel_speedup);
        Assert(false);
    }
}

void SetupOutputFolder() {
    if (output_override)
        wprintf("Overriding output folder <%s>...\n", output_folder_name.c_str());
    Assert(system(("mkdir -p " + output_folder_name).c_str()) == 0);
    Assert(system(("find " + output_folder_name + "/statistics -name \"*.config\" -type f | xargs rm -f").c_str()) == 0);
    // clean up dirs
    if (output_override && !is_simulation) {
        Assert(system(("rm -rf " + output_folder_name + "/include").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/src").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/bin").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/scripts").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/profiling_src").c_str()) == 0);
        Assert(system(("rm -f " + output_folder_name + "/main.cu").c_str()) == 0);
        Assert(system(("rm -f " + output_folder_name + "/main").c_str()) == 0);
    }
    // make dirs
    Assert(system(("mkdir -p " + output_folder_name + "/statistics").c_str()) == 0);
    // LRU visualization ////////////////////////////////////////////////////////////
    // Assert(system(("mkdir -p " + output_folder_name + "/lru_trace").c_str()) == 0);
    /////////////////////////////////////////////////////////////////////////////////
    if (!is_simulation) {
        Assert(system(("mkdir -p " + output_folder_name + "/include").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/src").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/bin").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/scripts").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/profiling_src").c_str()) == 0);
        Assert(system(("cp ./resources/cudadnnUtil.cuh " + output_folder_name + "/include/cudadnnUtil.cuh").c_str()) == 0);
        Assert(system(("cp ./resources/cudadnnUtil.cu " + output_folder_name + "/src/cudadnnUtil.cu").c_str()) == 0);
        Assert(system(("cp ./resources/Makefile " + output_folder_name + "/Makefile").c_str()) == 0);
        if (is_individual) {
            Assert(system(("cp ./resources/compileAndRunI.sh " + output_folder_name + "/scripts/compileAndRun.sh").c_str()) == 0);
        } else {
            Assert(system(("cp ./resources/compileAndRunW.sh " + output_folder_name + "/scripts/compileAndRun.sh").c_str()) == 0);
        }
    }
}

void loadWorkspaceSizes() {
    std::ifstream wok_f(workspace_size_file);
    Assert(wok_f.good());

    int kernel_id;
    string workspace_size_str;
    string unit;
    size_t workspace_size;
    iprintf("Loading workspace sizes from file <%s> for %d kernels\n",
            workspace_size_file.c_str(), kernel_list.size());
    for (int i = 0; i < kernel_list.size(); i++) {
        workspace_size_str = "";
        wok_f >> kernel_id >> workspace_size_str >> unit;
        Assert(kernel_id == i);
        Assert(workspace_size_str != "");
        Assert(unit == "B");
        workspace_size = std::stoull(workspace_size_str);
        Assert(workspace_size >= 0);
        if (workspace_size > 0) {
            kernel_list[i].workspace = new Tensor(workspace_size, false);
            kernel_list[i].outputs.insert(kernel_list[i].workspace);
            tensor_list.push_back(kernel_list[i].workspace);
        }
        
    }
    iprintf("Loading workspace sizes done\n\n", "");
}




void loadKernelTimes() {
    double GPU_frequency_Hz = GPU_frequency_GHz * pow(10, 9);
    
    std::ifstream orig_f(orig_kernel_time_file);
    std::ifstream pf_f(pf_kernel_time_file);
    std::ifstream inputpf_f(input_pf_kernel_time_file);
    Assert(orig_f.good());
    Assert(pf_f.good());
    Assert(inputpf_f.good());

    int kernel_num;
    string exe_time_ms_str; 
    string unit;
    // read in all the execution times
    long exe_time_cycle;
    double total_time = 0, pf_total_time = 0, input_pf_total_time = 0;
    unsigned long total_time_cycle = 0, pf_total_time_cycle = 0, input_pf_total_time_cycle = 0;
    iprintf("Loading kernel times from file <%s> and <%s> and <%s> for %d kernels\n",
            orig_kernel_time_file.c_str(), pf_kernel_time_file.c_str(), input_pf_kernel_time_file.c_str(), kernel_list.size());
    if (kernel_speedup != 1) {
        iprintf("Using kernel speedup of %.4fx\n", kernel_speedup);
    }
    for (int i = 0; i < kernel_list.size(); i++) {
        double delta_execution_time;
        // read in ideal execution time from file
        exe_time_ms_str.clear();
        orig_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        delta_execution_time = exe_time_cycle - exe_time_cycle / kernel_speedup;
        kernel_list[i].execution_cycles = exe_time_cycle - delta_execution_time;
        Assert(kernel_list[i].execution_cycles > 0);
        total_time += kernel_list[i].execution_cycles / GPU_frequency_Hz * 1000;
        total_time_cycle += exe_time_cycle;
        // read in input_pf execution time from file
        exe_time_ms_str.clear();
        inputpf_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        kernel_list[i].input_pf_execution_cycles = exe_time_cycle - delta_execution_time;
        if (kernel_list[i].input_pf_execution_cycles < kernel_list[i].execution_cycles)
            kernel_list[i].input_pf_execution_cycles = kernel_list[i].execution_cycles;
        // read in pf execution time from file
        exe_time_ms_str.clear();
        pf_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        kernel_list[i].pf_execution_cycles = exe_time_cycle - delta_execution_time;
        if (kernel_list[i].pf_execution_cycles < kernel_list[i].input_pf_execution_cycles)
            kernel_list[i].pf_execution_cycles = kernel_list[i].input_pf_execution_cycles;
        Assert(kernel_list[i].pf_execution_cycles > 0);
        Assert(exe_time_cycle > 0);
        pf_total_time += kernel_list[i].pf_execution_cycles / GPU_frequency_Hz * 1000;
        pf_total_time_cycle += exe_time_cycle;
        Assert(kernel_list[i].input_pf_execution_cycles > 0);
    }
    nprintf("Total time (Ideal): %f ms %lu cycles; (PF): %f ms %lu cycles\n", 
            total_time, total_time_cycle, pf_total_time, pf_total_time_cycle);
    // make sure kernel times file have no other entries left
    exe_time_ms_str = "";
    orig_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    // make sure pf kernel times file have no other entries left
    exe_time_ms_str = "";
    pf_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    // make sure inputpf kernel times file have no other entries left
    exe_time_ms_str = "";
    inputpf_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    iprintf("Loading kernel times done\n\n", "");
}

class RedirStdOut {
    public:
        RedirStdOut(std::string filename) {
            info_file = output_folder_name + "/statistics/" + filename;
            buffer.str("");
            old_cout_buf = std::cout.rdbuf();
            cout_buf = std::cout.rdbuf(buffer.rdbuf());
            printf("Saving %s\n", filename.c_str());
        }
        ~RedirStdOut() {
            std::ofstream fout(info_file.c_str());
            fout << buffer.str();
            fout.close();
            std::cout.rdbuf(old_cout_buf);
        }
    private:
        std::string info_file;
        std::stringstream buffer;
        std::streambuf *old_cout_buf;
        std::streambuf *cout_buf;
};

/* Function: main()
 * ----------------
 * Entry point to the entire program.  We parse the command line and turn
 * on any debugging flags requested by the user when invoking the program.
 * InitScanner() is used to set up the scanner.
 * InitParser() is used to set up the parser. The call to yyparse() will
 * attempt to parse a complete program from the input.
 */
int main(int argc, char *argv[]) {
    // Check config file argument
    if (argc == 1) {
        eprintf("Please specify a config file\n", "");
        Assert(false);
    }

    // Exit if config file does not exist
    std::ifstream config_file(argv[1]);
    if (!config_file.good()) {
        eprintf("Config file <%s> does not exist\n", argv[1]);
        Assert(false);
    }

    // Parse config file
    std::string line, command, value;
    printf("\nConfigs:\n");
    while (std::getline(config_file, line)) {
        std::stringstream ss(line);
        command.clear(); 
        value.clear();
        ss >> command >> value;

        if (command != "#" && command != "")
            printf("  %25s: <%s>\n", command.c_str(), value.c_str());

        // ===== General settings =====
        if (command == "output_folder")                 { output_folder_name = value; }
        else if (command == "output_override")          { output_override = std::stoi(value) != 0; }
        else if (command == "is_simulation")            { is_simulation = std::stoi(value) != 0; }
        else if (command == "is_transformer")           { is_transformer = std::stoi(value); }
        else if (command == "batch_size")               { batch_size = std::stoi(value); }
        else if (command == "input_H")                  { input_H = std::stoi(value); }
        else if (command == "input_W")                  { input_W = std::stoi(value); }
        else if (command == "num_threads")              { num_threads = std::stoi(value); }
        else if (command == "borden")                   { borden = std::stoi(value); }
        else if (command == "is_resnet")            { is_resnet = std::stoi(value); }
        else if (command == "is_inception")             { is_inception = std::stoi(value); }


        // ===== Model & profiling input files =====
        else if (command == "nn_model_input_file")      { nn_model_input_file = value; }
        else if (command == "orig_kernel_time_file")    { orig_kernel_time_file = value; }
        else if (command == "workspace_size_file")      { workspace_size_file = value; }
        else if (command == "input_pf_kernel_time_file"){ input_pf_kernel_time_file = value; }
        else if (command == "pf_kernel_time_file")      { pf_kernel_time_file = value; }
        else if (command == "stat_output_file")         { stat_output_file = value; }

        // ===== Simulation general =====
        else if (command == "is_UVM")                   { is_UVM = std::stoi(value) != 0; }
        else if (command == "use_prefetch")             { use_prefetch = std::stoi(value) != 0; }
        else if (command == "eviction_policy")          { eviction_policy_str = value; }

        // ===== System parameters =====
        else if (command == "system_latency_us")        { system_latency_us = std::stod(value); }
        else if (command == "CPU_PCIe_bandwidth_GBps")  { CPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "CPU_memory_line_GB")       { CPU_memory_line_GB = std::stod(value); }
        else if (command == "GPU_PCIe_bandwidth_GBps")  { GPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "GPU_memory_size_GB")       { GPU_memory_size_GB = std::stod(value); }
        else if (command == "GPU_frequency_GHz")        { GPU_frequency_GHz = std::stod(value); }
        else if (command == "GPU_malloc_uspB")          { GPU_malloc_uspB = std::stod(value); }
        else if (command == "GPU_free_uspB")            { GPU_free_uspB = std::stod(value); }
        else if (command == "SSD_PCIe_bandwidth_GBps")  { SSD_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "SSD_read_latency_us")      { SSD_read_latency_us = std::stod(value); }
        else if (command == "SSD_write_latency_us")     { SSD_write_latency_us = std::stod(value); }
        else if (command == "SSD_latency_us")           { SSD_latency_us = std::stod(value); }
        else if (command == "PCIe_latency_us")          { PCIe_latency_us = std::stod(value); }
        else if (command == "PCIe_batch_size_page")     { PCIe_batch_size_in_page = std::stoi(value); }
        else if (command == "delta_parameter")          { delta_parameter = std::stod(value); }

        // ===== Comments or empty =====
        else if (command == "#" || command == "")       {}
        else {
            eprintf("Error: Invalid config entry <%s>, aborting...\n", command.c_str());
            Assert(false);
        }
    }


    // Sanity check for GPU policies
    Assert((int) Simulator::GPUPageTable::EvcPolicy::DEEPUM != (int) Simulator::MigPolicy::DEEPUM);

    // Fix output folder path
    if (output_folder_name.back() == '/') output_folder_name.pop_back();
    printf("End configs\n\n");

    // Random seed
    srand(0);

    // Check output folder existence
    bool output_folder_exists = system(("test -d " + output_folder_name).c_str()) == 0;
    if (output_folder_exists && !output_override) {
        wprintf("Output folder <%s> exists\n", output_folder_name.c_str());
    }

    ParseCommandLine(argc, argv);
    SetupOutputFolder();

    // Model parsing and pre-pass
    if (is_transformer == 1) {
        transformer_parse(nn_model_input_file.c_str());
        transformer_op_datalow_pass(borden);
    } else {
        if (isatty(fileno(stdin))) {  // 如果当前输入不是管道输入
            if (nn_model_input_file.empty()) {
                eprintf("No input NN model in either stdin or config file\n", "");
                Assert(false);
            } else {
                std::ifstream nn_model(nn_model_input_file.c_str());
                if (!nn_model.good()) {
                    eprintf("Invalid input NN model specified in config file <%s>\n",
                            nn_model_input_file.c_str());
                    Assert(false);
                }
                freopen(nn_model_input_file.c_str(), "r", stdin);  // ⬅ 关键点：重定向
            }
        }
        std::printf("ready to init Scanner\n");
        InitScanner();
        InitParser();
        printf("ready to yyparse\n");
        yyparse();

        // std::ofstream log_file("layer_list.txt");
        // std::streambuf* cout_buf = std::cout.rdbuf(); // 保存原缓冲区
        // std::cout.rdbuf(log_file.rdbuf());            // 重定向到文件

        // // 打印层信息（所有 print_name 输出会进入 layer_list.txt）
        // for (int i = 0; i < forward_layers.size(); i++) {
        //     forward_layers[i]->print_name();
        // }

        // // 恢复 std::cout
        // std::cout.rdbuf(cout_buf);
        // log_file.close();

        std::printf("ready to layer analyse\n");
        layer_pre_pass_datasize();
        std::printf("layer pre done\n");
        layer_first_pass_dataflow();
        layer_second_pass_scheduling_kernels_ascend();
    }

    printf("\n");

    // Output tensor info (for Interval Time calculation)
    {
        RedirStdOut r("tensors.config");
        for (size_t i = 0; i < tensor_list.size(); i++)
            tensor_list[i]->print();
    }

    // Output layer info
    if (is_transformer == 1) {
        {
            RedirStdOut r("layers.config");
            for (size_t i = 0; i < forward_ops.size(); i++)
                forward_ops[i]->print();
        }
        transformer_scheduling_kernels();
    } else {
        {
            RedirStdOut r("layers.config");
            for (size_t i = 0; i < forward_layers.size(); i++)
                forward_layers[i]->print();
        }
    }
    //Output kernel info
    {
        RedirStdOut r("kernels_name.config");
        for (size_t i = 0; i < kernel_list.size(); i++)
            std::cout<<"Kernel ID: "<<kernel_list[i].kernel_id<<", "<< "Name: "<<print_ascendkerneltype_array[kernel_list[i].type_A]<<std::endl;
    }
    
    // Load kernel times (required for interval calculation)
    loadKernelTimes();

    // Tensor Interval Time calculation
    tensor_first_pass_liveness_analysis();
    tensor_second_pass_interval_formation();
    get_interval_time();

    // Export Interval Time to txt file
    // ==========  新增：完整的张量生命周期信息导出  ==========
    {
        std::string lifecycle_file = output_folder_name + "/tensor_lifecycle.txt";
        std::ofstream fout(lifecycle_file);
        if (!fout.is_open()) {
            eprintf("Cannot open file <%s> for writing\n", lifecycle_file.c_str());
            Assert(false);
        }

        /* 标题行，方便后续 python 画图 */
        fout << "#tensor_id  size_B  birth_kid  death_kid  "
                "hidden_interval_num  (start_kid,end_kid,duration_us) ...\n";

        for (Tensor* t : tensor_list) {
            if (t->is_global_weight)   // 全局权重不统计
                continue;

            /* 1. 基本生命段 */
            fout << t->tensor_id               << " "
                 << t->size_in_byte            << " "   // ← 改成现有字段
                 << t->live_interval[0]        << " "
                 << t->live_interval[1]        << " ";

            /* 2. 未激活段（hiding_interval） */
            fout << t->hidding_intervals.size();
            for (const Hidding_Interval* h : t->hidding_intervals) {
                // 转成微秒
                double dur_us = h->time_estimated;
                fout << " (" << h->kernelLevel_interval[0]
                     << ","  << h->kernelLevel_interval[1]
                     << ","  << dur_us << ")";
            }
            fout << "\n";
        }
        fout.close();
        printf("Tensor lifecycle exported to %s\n", lifecycle_file.c_str());
    }



    // Cleanup
    for (int i = 0; i < forward_layers.size(); i++)
        delete forward_layers[i];
    for (int i = 0; i < tensor_list.size(); i++)
        delete tensor_list[i];

    return (ReportError::NumErrors() == 0 ? 0 : -1);
}
