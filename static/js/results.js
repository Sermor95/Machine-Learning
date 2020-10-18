$(document).ready(function(){
    $('select').formSelect();
});
function filter(){
    var method = $('#filter-method option:selected').val();
    var criba = $('#filter-criba option:selected').val();
    var config = $('#config-id').val();

    console.log('mehtod-> '+method+' criba-> '+criba)
    $.ajax({
        type:'GET',
        url:'/analyze',
        data:'config='+config+'&method='+method+'&criba='+criba
    }).done(function(resp){
        $('#results').html($(resp).find('#results').html())
        M.toast({html: 'FILTER DONE'});
    });
}