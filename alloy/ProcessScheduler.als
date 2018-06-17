/* Models a Simple Process Scheduler
 */


// process id, unique to each process
sig ProcessID {}

// the possible states that a process can be in
abstract sig State {}
one sig Waiting, Active, Ready extends State {}


/* Q4, model the state and state invariant of a single processor scheduler
 */

// a single scheduler contains a set of processes
sig Scheduler {
  processes : set (ProcessID ->lone State)
}

// state invariant, only one process can be active at a given time
fact OneActiveProcess {
   all sch : Scheduler | lone sch.processes.Active
}


/* Q5, operation for adding new processes to a scheduler
 */
pred NewProcess[sch, sch': Scheduler, proc: ProcessID] {
   proc not in sch.processes.State
   and sch'.processes = sch.processes ++ (proc->Waiting)
}
run NewProcess


/* Q6, operation for shifting a process from state Waiting to Ready*/
pred Ready[sch, sch': Scheduler, proc: ProcessID] {
   proc in sch.processes.Waiting
   (no sch.processes.Active =>
      sch'.processes = sch.processes ++ (proc->Active)
   else
      sch'.processes = sch.processes ++ (proc->Ready)
   )
}
run Ready


/* Q7, takes an Active process and moves it to Waiting; if there is a Ready process, moves it
   to Active
 */
pred Swap[sch, sch': Scheduler] {
   some proc1, proc2: ProcessID | proc1 in sch.processes.Active
   and (proc2 in sch.processes.Ready =>
      sch'.processes = sch.processes ++ (proc1->Waiting) ++ (proc2->Active)
   else
      sch'.processes = sch.processes ++ (proc1->Waiting)
   )
}
run Swap
