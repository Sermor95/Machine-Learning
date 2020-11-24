$(document).ready(function(){
    $('#modal1').modal();
    $('select').formSelect();
});
function submitLaunch(){
    document.getElementById("preloader").style.display = "";
    var dataset = $('#dataset option:selected').val();
    var criba = $('#criba').val();
    var reduction = $('#reduction').val();
    var model = $('#model').val();
    var launchers = $('#launchers').val();
    var post = '{'+
            '"dataset": "'+dataset+'",'+
            '"criba": '+criba+','+
            '"reduction": '+reduction+','+
            '"model": "'+model+'",'+
            '"launchers": '+launchers+
        '}';
    $.ajax({
        url: '/feature-selection',
        type: 'POST',
        data: post,
        contentType: "application/json",
        dataType: 'json'
    }).always(function(resp){
        document.getElementById("preloader").style.display = "none";
        $('#configs').html($(resp.responseText).find('#configs').html());
        M.toast({html: 'Your launch was executed correctly'});
    });
    //     .fail(function (jqXHR, textStatus) {
    //     console.log('Error: '+jqXHR+', '+textStatus);
    // });
}

function analyzeConfig() {
    var dataset = $('#select-dataset option:selected').val();
    if (dataset != '') {
        document.getElementById("preloader").style.display = "";
        $.ajax({
            type: 'GET',
            url: '/analyze-config',
            data: 'dataset=' + dataset
        }).done(function (data) {
            resp = data.res
            $('#configs').html($(data.template).find('#configs').html());
            document.getElementById("preloader").style.display = "none";
            M.toast({html: 'FILTER DONE'});

            Highcharts.chart('chart-config', {

                chart: {
                    scrollablePlotArea: {
                        minWidth: 700,
                        minHeight: 700
                    }
                },

                data: {
                    columns: resp['series']
                },

                title: {
                    text: 'Average accuracy by CONFIG'
                },

                xAxis: {
                    categories: resp['categories']
                },

                yAxis: [{ // left y axis
                    title: {
                        text: null
                    },
                    labels: {
                        align: 'left',
                        x: 3,
                        y: 16,
                        format: '{value:.,0f}'
                    },
                    showFirstLabel: false
                }, { // right y axis
                    linkedTo: 0,
                    gridLineWidth: 0,
                    opposite: true,
                    title: {
                        text: null
                    },
                    labels: {
                        align: 'right',
                        x: -3,
                        y: 16,
                        format: '{value:.,0f}'
                    },
                    showFirstLabel: false
                }],

                legend: {
                    align: 'left',
                    verticalAlign: 'top',
                    borderWidth: 0
                },

                tooltip: {
                    shared: true,
                    crosshairs: true
                },

                plotOptions: {
                    series: {
                        cursor: 'pointer',
                        point: {
                            events: {
                                click: function (e) {
                                    hs.htmlExpand(null, {
                                        pageOrigin: {
                                            x: e.pageX || e.clientX,
                                            y: e.pageY || e.clientY
                                        },
                                        headingText: this.series.name,
                                        maincontentText: Highcharts.dateFormat('%A, %b %e, %Y', this.x) + ':<br/> ' +
                                            this.y + ' sessions',
                                        width: 200
                                    });
                                }
                            }
                        },
                        marker: {
                            lineWidth: 1
                        }
                    }
                },

                series: [{
                    name: 'All sessions',
                    lineWidth: 4,
                    marker: {
                        radius: 4
                    }
                }, {
                    name: 'New users'
                }]
            });

        });
    }


}

// const allRanges = document.querySelectorAll(".range-wrap");
// allRanges.forEach(wrap => {
//   const range = wrap.querySelector(".range");
//   const bubble = wrap.querySelector(".bubble");
//
//   range.addEventListener("input", () => {
//     setBubble(range, bubble);
//   });
//   setBubble(range, bubble);
// });
//
// function setBubble(range, bubble) {
//   const val = range.value;
//   const min = range.min ? range.min : 0;
//   const max = range.max ? range.max : 100;
//   const newVal = Number(((val - min) * 100) / (max - min));
//   bubble.innerHTML = val;
//
//   // Sorta magic numbers based on size of the native UI thumb
//   bubble.style.left = `calc(${newVal}% + (${8 - newVal * 0.15}px))`;
// }


function printValue(sliderID, textbox) {

    var x = document.getElementById(textbox);
    var y = document.getElementById(sliderID);
    if(sliderID=='launchers' && y.value==0)
        x.value = 1;
    else
        x.value = y.value;
}

// window.onload = function() { printValue('slider1', 'rangeValue1'); printValue('slider2', 'rangeValue2'); printValue('slider3', 'rangeValue3'); printValue('slider4', 'rangeValue4'); }


// function analyzeConfig(){
//     var dataset = $('#select-dataset option:selected').val();
//
//     $.ajax({
//         type:'GET',
//         url:'/analyze-config',
//         data:'dataset='+dataset
//     }).done(function(resp){
//         $('#configs').html($(resp).find('#configs').html())
//         M.toast({html: 'FILTER DONE'});
//
//         Highcharts.chart('chart-config', {
//
//             title: {
//                 text: 'Average accuracy by CONFIG'
//             },
//
//             yAxis: {
//                 title: {
//                     text: 'Accuracy'
//                 }
//             },
//
//             xAxis: {
//                 categories: resp['categories']
//             },
//
//             legend: {
//                 layout: 'vertical',
//                 align: 'right',
//                 verticalAlign: 'middle'
//             },
//
//             series: resp['series'],
//
//             responsive: {
//                 rules: [{
//                     condition: {
//                         maxWidth: 500
//                     },
//                     chartOptions: {
//                         legend: {
//                             layout: 'horizontal',
//                             align: 'center',
//                             verticalAlign: 'bottom'
//                         }
//                     }
//                 }]
//             }
//
//         });
//     });
//
//
//
// }

