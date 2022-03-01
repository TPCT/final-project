<?php
namespace core\form;

use core\Model;

class Form{
    public static function begin(string $action, string $method, string $name=''){
        echo "<form action='{$action}' method='{$method}'" . ($name ? "name='{$name}'>" : '>'); 
        return new Form();
    }

    public static function end(){
        echo "</form>";
    }

    public function field(Model $model, string $attribute,){
        return new Field($model, $attribute);
    }
}