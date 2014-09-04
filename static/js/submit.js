$(function () {

    // Globals
    var img = null;
    var imgarr = []
    var canvas = document.getElementById('my-canvas');
    var dropbox = document.getElementById('dropbox');
    var context = canvas.getContext('2d');
    var winwidth = $(window).width();
    var winheight = $(window).height();
    var IH = 1; // image height resized; arbitrarily init
    var IW = 1; // image height resized; arbitrarily init
    var MARGIN = 50;

    function drawStuff() {
	IW = canvas.width - MARGIN;
	var s = IW/img.width;

	IH = img.height*s;
	canvas.height = IH;

	context.drawImage(img,MARGIN/2,MARGIN/2,IW,IH);
    }  

    function getImagesMammo() {
	
	$.ajax({
	    type: "GET",
	    url: "/process_serve_mammo?imgfile="+curr_file,
	    data: {'message':'message'},
	    
	    success:function(resp) {
		if (resp.success > 0) {
		    
		    // Standard Implementation
		    img = new Image();
		    img.src = "data:image/png;base64," + resp.imagefile;
		    //img.src = "data:image/jpeg;base64," + resp.imagefile;

		    // New
		    img.onload = function () {
			resizeCanvas();
			imgarr = resp.imgarr;
			initCanvas();
		    }
		    
		    $("#AREA_D").html(resp.area_d);
		    $("#VOL_D").html(resp.volumetric_d);
		    $("#DCAT_A").html(resp.dcat_a);
		    $("#DCAT_V").html(resp.dcat_v);
		    $("#SIDE").html(resp.side);
		    $("#VIEW").html(resp.view);
		    //alert("SUCCESS for file: "+curr_file);
		} else {
		    alert("Sorry, there was an error processing the submitted image. Ensure that the mammogram file is of TIFF, JPEG, or DICOM format.");
		}

		$('#loading-indicator').hide();
		$('#dropbox-container').show();
		$('#results-container').show();
		$('#download-container').show();
		removeImages();
	    }
	});
    }

    function getImagesMRI() {
	
	$.ajax({
	    type: "GET",
	    url: "/process_serve_mri?imgfile="+curr_file,
	    data: {'message':'message'},
	    
	    success:function(resp) {
		if (resp.success > 0) {
		    
		    // Experimental Implementation
		    img = new Image();
		    img.src = "data:image/png;base64," + resp.imagefile;
		    img.onload = function () {
			resizeCanvas();
			imgarr = resp.imgarr;
			initCanvas();
		    }
		    $("#AREA_D").html(resp.area_d);
		    $("#VOL_D").html(resp.volumetric_d);
		    $("#DCAT_A").html(resp.dcat_a);
		    $("#DCAT_V").html(resp.dcat_v);
		    $("#SIDE").html(resp.side);
		    $("#VIEW").html(resp.view);
		} else {
		    alert("Sorry, there was an error processing one of the submitted MRI slices.");
		}

		$('#loading-indicator').hide();
		$('#dropbox-container').show();
		$('#results-container').show();
		$('#download-container').show();
		removeImages();
	    }
	});
    }

   $("#processButtonMammo").on("click",function(e) {
        e.preventDefault();

	if (curr_file.length < 1) {
	    alert("No mammogram was found. Please drag and drop a mammogram file into the web page drop box.");
	} else {
	    
	    $('#loading-indicator').show();
	    $('#dropbox-container').hide();
	    $('#results-container').hide();
	    $('#download-container').hide();	    
	    getImagesMammo();
	}
    }); 

    $("#processButtonMRI").on("click",function(e) {
        e.preventDefault();

	if (curr_file.length < 1) {
	    alert("No mammogram was found. Please drag and drop a mammogram file into the web page drop box.");
	} else {
	    
	    $('#loading-indicator').show();
	    $('#dropbox-container').hide();
	    $('#results-container').hide();
	    $('#download-container').hide();	    
	    getImagesMRI();
	}
    }); 

    function removeImages () {
	var dropbox = $('#dropbox'),
	message = $('.message', dropbox);
	$('.preview').remove();
	message.show();
	curr_file = [];
    }
    
    function resizeCanvas () {
        var prev_width = canvas.width;
        //var new_width = window.innerWidth;
        var new_width = window.innerWidth*0.6;

	canvas.width = new_width + 25; //arbitrary buffer
	//canvas.height = IH;
	canvas.height = canvas.width; // make it square
	//canvas.height = canvas.height * (new_width/prev_width);	
        drawStuff(); 
    }
   
    function initCanvas() {

	// Display the first image
	myimg = document.createElement('img');
        myimg.src = imgarr[0];

	// We need to wait until the image is loaded before we can get its width and height
        myimg.onload = function() {
	    //var IW = myimg.width;
	    //var IH = myimg.height;	    

	     $("#my-canvas").mooscan({
    		images:imgarr,
		imgx:MARGIN/2,
		imgy:MARGIN/2,
		niw:IW,
		nih:IH,
		winwidth:canvas.width,
		winheight:canvas.height,
		oiw:IW,
		oih:IH
	    });
	}
    }

    // resize the canvas to fill browser window dynamically
    //window.addEventListener('resize', resizeCanvas, false); 

});