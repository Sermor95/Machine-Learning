$(document).ready(function(){
    $('#modal1').modal();
});
function submitLaunch(){
    var dataset = $('#dataset').val();
    var criba = $('#criba').val();
    var limit = $('#limit').val();
    var pearson_base = $('#pearson_base').val();
    var ohe = $('#ohe').val();
    var cat_feat = $('#cat_feat').val();
    var post = '{'+
            '"dataset": "'+dataset+'",'+
            '"criba": '+criba+','+
            '"limit": '+limit+','+
            '"pearson_base": '+pearson_base+','+
            '"ohe": "'+ohe+'",'+
            '"categorical_features": '+cat_feat+
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