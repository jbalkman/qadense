$(function(){
    
    var dropbox = $('#dropbox'),
    message = $('.message', dropbox);
    curr_file = []; // set global variable to null and flag error if process button is pressed prior to uploading a mammogram
 
    //var canvas = document.getElementById("myCanvas");
    //var context = canvas.getContext("2d");

    dropbox.filedrop({
	paramname: 'file',
	maxfiles: 4,
    	maxfilesize: 30,
	url: '/upload',
	uploadFinished:function(i, file, response) {
	    $.data(file).addClass('done');
	    /*alert("Upload finished: "+response.file);
	    var img = new Image();
	    img.src = "/serve_img?file="+response.file;
	    img.onload = function () {
		alert("Image source: "+img.src);
		context.drawImage(img, 0, 0, 100, 100);
		alert("Context: "+context);
	    }*/
	    curr_file = response.file;

	    //alert("My curr file = "+curr_file);

	    //alert("Current file after drop: "+curr_file);
	},
	
    	error: function(err, file) {
	    switch(err) {
	    case 'BrowserNotSupported':
		showMessage('Your browser does not support HTML5 file uploads!');
		break;
	    case 'TooManyFiles':
		alert('Too many files! Please select ' + this.maxfiles + ' at most!');
		break;
	    case 'FileTooLarge':
		alert(file.name + ' is too large! The size is limited to ' + this.maxfilesize + 'MB.');
		break;
	    default:
		break;
	    }
	},
	
	beforeEach: function(file){
	    if(!(file.type.match(/^image\//) || file.type.match(/dicom/))){
		alert('Only images are allowed!');
		return false;
	    }
	},
	
	uploadStarted:function(i, file, len){
	    createImage(file);
	},
	
	progressUpdated: function(i, file, progress) {
	    $.data(file).find('.progress').width(progress);
	}
    });
    
    var template = '<div class="preview">'+
	'<span class="imageHolder">'+
	'<img />'+
	'<span class="uploaded"></span>'+
	'</span>'+
	'<div class="progressHolder">'+
	'<div class="progress"></div>'+
	'</div>'+
	'</div>';
    
    function createImage(file){
	
	    var preview = $(template), 
	    image = $('img', preview);
	    
	    var reader = new FileReader();
	    
            image.width = 100;
	    image.height = 100;
	    
	    reader.onload = function(e){
		image.attr('src',e.target.result);
	    };
	    
	    reader.readAsDataURL(file);
	    
	    message.hide();
	    preview.appendTo(dropbox);
	    
	    $.data(file,preview);
	}
    
    function showMessage(msg){
	message.html(msg);
    }
    
});