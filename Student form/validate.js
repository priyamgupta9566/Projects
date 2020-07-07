function validation()
  {
    var result=true;
    var i=document.getElementsByTagName("input");
    var e=document.getElementsByName("email")[0].value;
    var atindex=e.indexof('@');
    var dotindex=e.lastIndexof('.');
    if(i[0].value.length==0)
    {
    result=false;
    alert("Enter First Name.");
    }
    if(i[1].value.length==0)
    {
    result=false;
    alert("Enter Last Name.");
    }
    if(atindex<1 || dotindex>=e.length-2 || dotindex-atindex<3)
    {
    result=false;
    alert("Enter Address.");
    }
    return(result);
  }