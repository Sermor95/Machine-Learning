$(document).ready(function(){
    $('select').formSelect();
    analyzeConfig()
});
function filter(){
    var method = $('#filter-method option:selected').val();
    var criba = $('#filter-criba option:selected').val();
    var config = $('#config-id').val();

    console.log('mehtod-> '+method+' criba-> '+criba)
    $.ajax({
        type:'GET',
        url:'/analyze-results',
        data:'config='+config+'&method='+method+'&criba='+criba
    }).done(function(resp){
        $('#results').html($(resp).find('#results').html())
        M.toast({html: 'FILTER DONE'});
    });
}

function analyzeConfig(){
    var config_id = $('#config-id').val();

    $.ajax({
        type:'GET',
        url:'/analyze-result',
        data:'config='+config_id
    }).done(function(resp){
        $('#configs').html($(resp).find('#configs').html())
        M.toast({html: 'FILTER DONE'});

        Highcharts.chart('chart-result', {

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