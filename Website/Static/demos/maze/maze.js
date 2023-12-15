data_r = fetch("./maze.json")
.then(response => {
    return response.json();
});
console.log(data_r);

canvas = document.getElementById('maze');
ctx = canvas.getContext('2d');
gen_display = document.getElementById('gen');

//graph contexts
fit_ctx = document.getElementById('fit_graph').getContext('2d');
alive_ctx = document.getElementById('alive_graph').getContext('2d');
dist_ctx = document.getElementById('dist_graph').getContext('2d');
moves_ctx = document.getElementById('moves_graph').getContext('2d');

// chart vars
var fit_chart = new Chart(fit_ctx);
var alive_chart = new Chart(alive_ctx);
var dist_chart = new Chart(dist_ctx);
var moves_chart = new Chart(moves_ctx);

// create maze
var scale_size = 30;
var maze_size = 12;
ctx.canvas.height = scale_size * maze_size;
ctx.canvas.width = scale_size * maze_size;
ctx.scale(scale_size, scale_size);

// destroy charts
function destroy_charts() {
    fit_chart.destroy();
    alive_chart.destroy();
    dist_chart.destroy();
    moves_chart.destroy();
}


// generate new charts
function generate_new_charts() {

    fit_chart = new Chart(fit_ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Maximum Fitness of Each Generation',
                data: [],
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
    
    });
    alive_chart = new Chart(alive_ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Proportion of Surviving Agents of Each Generation',
                data: [],
                fill: false,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
    });
    dist_chart = new Chart(dist_ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Minimum Distance From Goal of Each Generation',
                data: [],
                fill: false,
                borderColor: 'rgb(255, 159, 64)',
                tension: 0.1
            }]
        },
    });
    moves_chart = new Chart(moves_ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Moves Taken to Reach Goal or Die of Each Generation',
                data: [],
                fill: false,
                borderColor: 'rgb(153, 102, 255)',
                tension: 0.1
            }]
        },
    });
}

destroy_charts();
generate_new_charts();

function draw_maze() {
    // add border
    for (i = 0; i < maze_size; i++) {
        // top
        ctx.fillRect(i, 0, 1, 1);
        //bottom
        ctx.fillRect(i, maze_size-1, 1, 1);
        //left
        ctx.fillRect(0, i, 1, 1);
        //right
        ctx.fillRect(maze_size-1, i, 1, 1);
    }
    // add barrier
    for (i = 0; i< maze_size-6; i++) {
        for (j = 0; j< maze_size-6; j++) {
            ctx.fillRect(i+3, j+3, 1, 1);
        }
    }
    // add red and green end and starts
    ctx.fillStyle = "#FF0000";
    ctx.fillRect(1, 10, 1, 1)
    ctx.fillStyle = "#00FF00";
    ctx.fillRect(10, 1, 1, 1)
    ctx.fillStyle = "#000000";

}

function draw_circle(x, y) {
    // draw circle
    ctx.fillStyle = "#0000FF";
    ctx.beginPath();
    ctx.arc(x+0.5, y+0.5, 0.2, 0, 2 * Math.PI);
    ctx.fill()
    ctx.fillStyle = "#000000";
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

draw_maze();
draw_circle(10, 1);


async function preset1(preset) {
    const data = await data_r;
    preset_data = data[preset];
    // populate graphs first
    // destroy charts
    destroy_charts();
    // gen new charts
    generate_new_charts();
    // add data to new chart
    for (i = 0; i < preset_data['gen_nums'].length; i++) {
        fit_chart.data.labels.push(preset_data['gen_nums'][i]);
        alive_chart.data.labels.push(preset_data['gen_nums'][i]);
        dist_chart.data.labels.push(preset_data['gen_nums'][i]);
        moves_chart.data.labels.push(preset_data['gen_nums'][i]);
        // add data to each cart
        fit_chart.data.datasets.forEach((dataset) => {
            dataset.data.push(preset_data['max_fit'][i]);
        });
        alive_chart.data.datasets.forEach((dataset) => {
            dataset.data.push(preset_data['alive_prop'][i]);
        });
        dist_chart.data.datasets.forEach((dataset) => {
            dataset.data.push(preset_data['min_dist'][i]);
        });
        moves_chart.data.datasets.forEach((dataset) => {
            dataset.data.push(preset_data['max_moves'][i]);
        });
        // update charts
        fit_chart.update();
        alive_chart.update();
        dist_chart.update();
        moves_chart.update();
    }


    // ok just move the circle 3 to the left
    // animation loop
    // data: 3d arr: generation, x y pair of each move 
    agent_moves = preset_data['agent_moves'];
    //console.log(agent_moves[0][0].length);
    ctx.clearRect(0, 0, maze_size, maze_size);
    draw_maze();
    draw_circle(10, 1);
    // for each generation
    for (g = 0; g < agent_moves.length; g++) {

        ctx.clearRect(0, 0, maze_size, maze_size);
        draw_maze();
        draw_circle(10, 1);

        gen_display.innerHTML = g;
        // for each move in xy pair
        for (m = 0; m < agent_moves[g][0].length; m++) {
             //clear canvas
            ctx.clearRect(0, 0, maze_size, maze_size);
            ctx.save();
            // draw maze
            draw_maze();
            // draw circle
            x = agent_moves[g][0][m];
            y = agent_moves[g][1][m];
            draw_circle(x, y);
            ctx.restore();
            await sleep(50);
        }
    }
    console.log('done')
    ctx.clearRect(0, 0, maze_size, maze_size);
    draw_maze();
    draw_circle(10, 1);
}