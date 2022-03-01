<?php

namespace core;

class Request{
    public function method(){
        return $_SERVER['REQUEST_METHOD'];
    }

    public function path(){
        $path = $_SERVER['REQUEST_URI'] ?? '/';
        $mark_position = strpos($path, '?');
        if ($mark_position !== False)
            $path = substr($path, 0, $mark_position);
        return $path;
    }

    public function isGet(){
        return $this->method() === 'GET';
    }

    public function isPost(){
        return $this->method() === 'POST';
    }

    public function body(){
        $body = array();
        if ($this->isPost()){
            foreach($_POST as $key => $value){
                if (is_array($value)){
                    $body[$key] = [];
                    foreach ($value as $valueItem){
                        $body[$key][] = \filter_var($valueItem, \FILTER_SANITIZE_SPECIAL_CHARS);
                    }
                }
                else
                    $body[$key] = filter_var($value, \FILTER_SANITIZE_SPECIAL_CHARS);
            }
        }

        foreach($_GET as $key => $value){
            if (is_array($value)){
                $body[$key] = [];
                foreach ($value as $valueItem){
                    $body[$key][] = \filter_var($valueItem, FILTER_SANITIZE_SPECIAL_CHARS);
                }
            }
            else
                $body[$key] = \filter_input(\INPUT_GET, $key, \FILTER_SANITIZE_SPECIAL_CHARS);
        }
        return $body;
    }
}