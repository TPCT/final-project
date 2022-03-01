<?php
namespace core\Exceptions;

use Exception;

class RequiredFileNotFound extends Exception{
    public function __construct(string $path){
        $this->message = "The Required File {$path} is not found\n";
    }
}