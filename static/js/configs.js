$(document).ready(function(){
    $('#modal1').modal();
    $('select').formSelect();
    $('#select-attribute').change(function (){
        var options = [];
        var values = [];
        var opt = $('#select-attribute').val();
        $.ajax({
                async: false,
                type: 'GET',
                url: '/get_distinct_from_config',
                data: 'attr='+opt
            }).done(function (resp) {
                values = resp.values
            });
            values.forEach(v => {
	    		options += '<option value="'+v+'">'+v+'</option>';
	    	});

            $('#options').html("<select id='select-value' required>"+
	    		  options+
				"</select><label>Value</label>");
            $('#select-value').formSelect();
    })
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
}

function analyzeConfig() {
    var dataset = $('#select-dataset option:selected').val();
    var attribute = $('#select-attribute option:selected').val();
    var value = $('#select-value option:selected').val();
    $('#charts-generated').empty();
    if (dataset != '') {
        document.getElementById("preloader").style.display = "";
        $.ajax({
            type: 'GET',
            url: '/analyze-config',
            data: 'dataset=' + dataset + '&attribute='+attribute + '&value='+value
        }).done(function (data) {
            var elem = data['res'];
            $('#configs').html($(data.template).find('#configs').html());
            document.getElementById("preloader").style.display = "none";
            M.toast({html: 'FILTER DONE'});

            for(i=0;i<elem.length;i++) {
                var chartCustom = elem[i];

                if(data['is_base']==true){
                    var chartTimeCustom = data['times'][i];
                    $('#charts-generated').append("<figure class=highcharts-figure>" +
                    "   <div class='row'>"    +
                    // "        <div id=chart-config"+i+" class='col s8'></div>" +
                    "        <div id=chart-config-time"+i+"></div>" +
                    "    </div>"+
                    "    </figure>");

                    Highcharts.chart('chart-config-time'+i, {
                        chart: {
                            type: 'column'
                        },
                        title: {
                            text: chartCustom['title']
                        },
                        xAxis: {
                            categories: chartTimeCustom['methods'],
                            crosshair: true
                        },
                        yAxis: {
                            min: 0,
                            title: {
                                text: 'Time (seconds)'
                            }
                        },
                        tooltip: {
                            headerFormat: '<span style="font-size:10px">{point.key}</span><table>',
                            pointFormat: '<tr><td style="color:{series.color};padding:0">{series.name}: </td>' +
                                '<td style="padding:0"><b>{point.y:.1f} s</b></td></tr>',
                            footerFormat: '</table>',
                            shared: true,
                            useHTML: true
                        },
                        plotOptions: {
                            column: {
                                pointPadding: 0.2,
                                borderWidth: 0
                            }
                        },
                        series: [{
                            name: 'Without criba',
                            data: chartTimeCustom['twoc']

                        }, {
                            name: 'With criba',
                            data: chartTimeCustom['twc']

                        }]
                    });

                }else{
                    $('#charts-generated').append("<figure class=highcharts-figure>" +
                    "        <div id=chart-config"+i+"></div>" +
                    "    </figure>");

                    Highcharts.chart('chart-config'+i, {

                        chart: {
                            scrollablePlotArea: {
                                minWidth: 700,
                                minHeight: 700
                            }
                        },

                        data: {
                            columns: chartCustom['chartParam']['series']
                        },

                        title: {
                            text: chartCustom['title']
                        },

                        xAxis: {
                            categories: chartCustom['chartParam']['categories']
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
                }

            }
        });
    }
}

function printValue(sliderID, textbox) {

    var x = document.getElementById(textbox);
    var y = document.getElementById(sliderID);
    if(sliderID=='launchers' && y.value==0)
        x.value = 1;
    else
        x.value = y.value;
}



