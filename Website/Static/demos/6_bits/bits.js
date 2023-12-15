data_r = fetch("./bits.json")
.then(response => {
    return response.json();
});
console.log(data_r);

preset1_btn = document.getElementById('preset1');
canvas = document.getElementById('bits');
ctx = canvas.getContext('2d');

best_canvas = document.getElementById('bests');
best_ctx = best_canvas.getContext('2d');


const graph_ctx = document.getElementById('graph').getContext('2d');

function generate_new_chart() {
    return new Chart(graph_ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Maximum Sum Per Generation',
                data: [],
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
    });
}

myChart = generate_new_chart();
//console.log(myChart.data)


async function preset1(preset) {
    // cp this code each time we want to access data
    const data = await data_r;
    var scale_size = 30;
    preset_data = data[preset];
    console.log(preset_data);
    // resize canvas for init population
    init_pop = preset_data['init_pop'];
    arr_len = init_pop[0].length;
    pop_size = init_pop.length;
    ctx.canvas.height = pop_size * scale_size;
    ctx.canvas.width = arr_len * scale_size;
    ctx.scale(scale_size, scale_size);
    // add border around canvas
    ctx.rect(0, 0, arr_len, pop_size);

    //ctx.fillRect(1, 0, 1, 1);
    // fill pop canvas
    for (i = 0; i < pop_size; i++){
        for (j = 0; j < arr_len; j++) {
            if (init_pop[i][j]) {
                ctx.fillRect(j, i, 1, 1);
            }
        }
    }

    // rem all data from chart
    myChart.data.labels.pop();
    myChart.data.datasets.forEach((dataset) => {
        dataset.data.pop();
    });
    myChart.update();
    // destroy and create new chart
    myChart.destroy();
    myChart = generate_new_chart();
    // add data to chart
    for (i = 0; i < preset_data['gen_nums'].length; i++) {
        myChart.data.labels.push(preset_data['gen_nums'][i]);
        myChart.data.datasets.forEach((dataset) => {
            dataset.data.push(preset_data['best_sums'][i]);
        });
        myChart.update();
    }
    // add best bits to chart
    best_arrs = preset_data['best_arrs']

    best_arr_len = best_arrs[0].length;
    best_arr_size = best_arrs.length;
    best_ctx.canvas.height = best_arr_size * scale_size;
    best_ctx.canvas.width = best_arr_len * scale_size;
    best_ctx.scale(scale_size, scale_size);
    // add border around canvas
    best_ctx.rect(0, 0, best_arr_len, best_arr_size);

    //ctx.fillRect(1, 0, 1, 1);
    // fill pop canvas
    for (i = 0; i < best_arr_size; i++){
        for (j = 0; j < best_arr_len; j++) {
            if (best_arrs[i][j]) {
                best_ctx.fillRect(j, i, 1, 1);
            }
        }
    }
    

}