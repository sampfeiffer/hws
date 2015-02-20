#ifndef USAGE_INCLUDED
#define USAGE_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

// This header file grabs information about the system memory and cpu usage.
// Found the majority of this code online.

static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;

void init(){
    FILE* file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %Ld %Ld %Ld %Ld", &lastTotalUser, &lastTotalUserLow,
        &lastTotalSys, &lastTotalIdle);
    fclose(file);
}

double getCurrentValue(){
    double percent;
    FILE* file;
    unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;


    file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %Ld %Ld %Ld %Ld", &totalUser, &totalUserLow,
        &totalSys, &totalIdle);
    fclose(file);


    if (totalUser < lastTotalUser || totalUserLow < lastTotalUserLow ||
        totalSys < lastTotalSys || totalIdle < lastTotalIdle){
        //Overflow detection. Just skip this value.
        percent = -1.0;
    }
    else{
        total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) +
            (totalSys - lastTotalSys);
        percent = total;
        total += (totalIdle - lastTotalIdle);
        percent /= total;
        percent *= 100;
    }


    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;


    return percent;
}

struct sysinfo memInfo;

std::string get_info()
{
    std::stringstream info_stream;
    sysinfo (&memInfo);
    long long total_virtual_mem = memInfo.totalram;
    long long virtual_mem_used = total_virtual_mem - memInfo.freeram;
    long long total_phys_mem = memInfo.totalram;
    long long phys_mem_used = memInfo.totalram - memInfo.freeram;

    //Add and multiply other values in next statement to avoid int overflow on right hand side...
    total_virtual_mem += memInfo.totalswap;
    total_virtual_mem *= memInfo.mem_unit;
    virtual_mem_used += memInfo.totalswap - memInfo.freeswap;
    virtual_mem_used *= memInfo.mem_unit;
    total_phys_mem *= memInfo.mem_unit;
    phys_mem_used *= memInfo.mem_unit;


    info_stream << "Total virtual memory: " << total_virtual_mem << "\n"
                << "Total virtual memory used: " << virtual_mem_used << "\n"
                << "Total physical memory: " << total_phys_mem << "\n"
                << "Total physical memory used: " << phys_mem_used << "\n"
                << "Total cpu in use: " << getCurrentValue() << "%\n"
                << "-----------------------------------------\n\n";

    return info_stream.str();
}

#endif // USAGE_INCLUDED
