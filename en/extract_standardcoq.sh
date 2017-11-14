#/bin/sh
ccg="/Users/yanakahitomi/new/verypluming/ccg2lambda/results/"


for f in `cat difficult_sick_trial_id.txt`; do
    #num=($(cat ${ccg}sick_trial_${f}.html|grep -n "echo"|awk -F: '{print $1}'))
    #start=${num[0]}
    #end=$((${num[1]}-3))
    #coq_script=$(sed -n ${start},${end}p ${ccg}sick_trial_${f}.html)
    #format_coq_script=${coq_script/Qed./Qed. repeat nltac_base.}
    #eval "${format_coq_script}" > ${ccg}sick_trial_${f}.coq.txt
    subnum=$(cat ${ccg}sick_trial_${f}.coq.txt|grep -n "Error: Attempt to save an incomplete proof"|awk -F: '{print $1}')
    subgoalnum=$(($subnum+1))
    tail -n +${subgoalnum} ${ccg}sick_trial_${f}.coq.txt
    printf "\n"
    #rm ${ccg}sick_trial_${f}.coq.txt
done

#while read line
#do
#    eval "${line}"
#done < ${1}