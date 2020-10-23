$(document).ready(function(){
    $('#modal1').modal();
    $('select').formSelect();
});
function submitLaunch(){
    var dataset = $('#dataset').val();
    var criba = $('#criba').val();
    var reduction = $('#reduction').val();
    var post = '{'+
            '"dataset": "'+dataset+'",'+
            '"criba": '+criba+','+
            '"reduction": '+reduction+
        '}';
    $.ajax({
        url: './feature-selection',
        type: 'POST',
        data: post,
        contentType: "application/json",
        dataType: 'json'
    }).done(function(data){
        M.toast({html: 'Your launch was executed correctly'});
    });
}

function analyzeConfig(){
    var dataset = $('#select-dataset option:selected').val();

    $.ajax({
        type:'GET',
        url:'/analyze-config',
        data:'dataset='+dataset
    }).done(function(resp){
        $('#configs').html($(resp).find('#configs').html())
        M.toast({html: 'FILTER DONE'});

        Highcharts.chart('chart-config', {

            chart: {
                scrollablePlotArea: {
                    minWidth: 700,
                    minHeight: 700
                }
            },

            data: {
            columns: resp['series']},

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

