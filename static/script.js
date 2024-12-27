var optionsSets = [
    ["itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering","chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting"
,"vomiting","burning_micturition","spotting_urination","fatigue","weight_gain","anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness","lethargy"
,"patches_in_throat","irregular_sugar_level","cough","high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache"
,"yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","constipation","abdominal_pain","diarrhoea","mild_fever"
,"yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise"
,"blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs"
,"fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising"
,"obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger"
,"extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck","swelling_joints",
"movement_stiffness","spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort",
"foul_smell_ofurine","continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)","depression"
,"irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes","increased_appetite"
,"polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections"
,"coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","blood_in_sputum","prominent_veins_on_calf","palpitations"
,"painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister"
,"red_sore_around_nose","yellow_crust_ooze","prognosis"],
];
var symptom_weight={"itching":1,"skin_rash":3,"nodal_skin_eruptions":4,"continuous_sneezing":4,"shivering":5,"chills":3,"joint_pain":3,"stomach_pain":5,"acidity":3,"ulcers_on_tongue":4,"muscle_wasting":3
,"vomiting":5,"burning_micturition":6,"spotting_urination":6,"fatigue":4,"weight_gain":3,"anxiety":4,"cold_hands_and_feets":5,"mood_swings":3,"weight_loss":3,"restlessness":5,"lethargy":2
,"patches_in_throat":6,"irregular_sugar_level":5,"cough":4,"high_fever":7,"sunken_eyes":3,"breathlessness":4,"sweating":3,"dehydration":4,"indigestion":5,"headache":3
,"yellowish_skin":3,"dark_urine":4,"nausea":5,"loss_of_appetite":4,"pain_behind_the_eyes":4,"back_pain":3,"constipation":4,"abdominal_pain":4,"diarrhoea":6,"mild_fever":5
,"yellow_urine":4,"yellowing_of_eyes":4,"acute_liver_failure":6,"fluid_overload":6,"swelling_of_stomach":7,"swelled_lymph_nodes":6,"malaise":6
,"blurred_and_distorted_vision":5,"phlegm":5,"throat_irritation":4,"redness_of_eyes":5,"sinus_pressure":4,"runny_nose":5,"congestion":5,"chest_pain":7,"weakness_in_limbs":7
,"fast_heart_rate":5,"pain_during_bowel_movements":5,"pain_in_anal_region":6,"bloody_stool":5,"irritation_in_anus":6,"neck_pain":5,"dizziness":4,"cramps":4,"bruising":4
,"obesity":4,"swollen_legs":5,"swollen_blood_vessels":5,"puffy_face_and_eyes":5,"enlarged_thyroid":6,"brittle_nails":5,"swollen_extremeties":5,"excessive_hunger":4
,"extra_marital_contacts":5,"drying_and_tingling_lips":4,"slurred_speech":4,"knee_pain":3,"hip_joint_pain":2,"muscle_weakness":2,"stiff_neck":4,"swelling_joints":5,
"movement_stiffness":5,"spinning_movements":6,"loss_of_balance":4,"unsteadiness":4,"weakness_of_one_body_side":4,"loss_of_smell":3,"bladder_discomfort":4,
"foul_smell_ofurine":5,"continuous_feel_of_urine":6,"passage_of_gases":5,"internal_itching":4,"toxic_look_(typhos)":5,"depression":3
,"irritability":2,"muscle_pain":2,"altered_sensorium":2,"red_spots_over_body":3,"belly_pain":4,"abnormal_menstruation":6,"dischromic_patches":6,"watering_from_eyes":4,"increased_appetite":5
,"polyuria":4,"family_history":5,"mucoid_sputum":4,"rusty_sputum":4,"lack_of_concentration":3,"visual_disturbances":3,"receiving_blood_transfusion":5,"receiving_unsterile_injections":2
,"coma":7,"stomach_bleeding":6,"distention_of_abdomen":4,"history_of_alcohol_consumption":5,"fluid_overload":4,"blood_in_sputum":5,"prominent_veins_on_calf":6,"palpitations":5
,"painful_walking":2,"pus_filled_pimples":2,"blackheads":2,"scurring":2,"skin_peeling":3,"silver_like_dusting":2,"small_dents_in_nails":2,"inflammatory_nails":2,"blister":4
,"red_sore_around_nose":2,"yellow_crust_ooze":3,"prognosis":5}
function addDropdown() {
    var newDropdownContainer = document.createElement("div");
    newDropdownContainer.classList.add("dropdown-container");

    var newSearchContainer = document.createElement("div");
    newSearchContainer.classList.add("search-container");

    var newSearchDropdown = document.createElement("input");
    newSearchDropdown.type = "text";
    newSearchDropdown.classList.add("search-dropdown");
    newSearchDropdown.oninput = function () {
        filterOptions(newSearchDropdown);
    };

    var newDropdownList = document.createElement("ul");
    newDropdownList.classList.add("dropdown-list");

    newSearchContainer.appendChild(newSearchDropdown);
    newSearchContainer.appendChild(newDropdownList);

    newDropdownContainer.appendChild(newSearchContainer);
    document.getElementById("dropdownContainer").appendChild(newDropdownContainer);

    optionsSets.push(["itching","skin_rash,nodal_skin_eruptions","continuous_sneezing","shivering","chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting"
    ,"vomiting","burning_micturition","spotting_urination","fatigue","weight_gain","anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness","lethargy"
    ,"patches_in_throat","irregular_sugar_level","cough","high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache"
    ,"yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","constipation","abdominal_pain","diarrhoea","mild_fever"
    ,"yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise"
    ,"blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs"
    ,"fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising"
    ,"obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger"
    ,"extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck","swelling_joints",
    "movement_stiffness","spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort",
    "foul_smell_ofurine","continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)","depression"
    ,"irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes","increased_appetite"
    ,"polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections"
    ,"coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","blood_in_sputum","prominent_veins_on_calf","palpitations"
    ,"painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister"
    ,"red_sore_around_nose","yellow_crust_ooze","prognosis"]);
}

function filterOptions(input) {
    var filter = input.value.toUpperCase();
    var dropdownList = input.nextElementSibling;
    var options = optionsSets[0];

    dropdownList.style.display = "block";
    dropdownList.innerHTML = "";

    var filteredOptions = options.filter(function (option) {
        return option.toUpperCase().includes(filter);
    });

    filteredOptions.forEach(function (option) {
        var listItem = document.createElement("li");
        listItem.textContent = option;
        listItem.onclick = function () {
            input.value = option;
            dropdownList.style.display = "none";
        };
        dropdownList.appendChild(listItem);
    });
}

function submitForm() {
    var selectedOptions = [];
    var searchInputs = document.querySelectorAll(".search-dropdown");

    searchInputs.forEach(function (input) {
        selectedOptions.push(input.value);
    });

    var weight = calculateWeight(selectedOptions);
    $.ajax({
        type: 'POST',
        contentType: 'application/json;charset=UTF-8',
        url: '/classify',
        data:  JSON.stringify({ weights: weight }),  // Send data as an object
        success: function (response) {
            console.log(response)
            displayResult(response);
        },
        error: function (error) {
            console.error('Error during classification:', error);
        }
    });
}

function displayResult(result) {
    var resultContainer = document.getElementById("result-container");
    resultContainer.innerHTML = "<p>PREDICTED DISEASE: " + result.toString()+ "</p>";
}

function calculateWeight(selectedOptions) {
    var weight = [];
    selectedOptions.forEach(function (symptom) {
        if (symptom_weight.hasOwnProperty(symptom)) {
            weight.push(symptom_weight[symptom]);
        }
    });
    while (weight.length < 17) {
        weight.push(0);
    }
    return weight;
}
