<?php
namespace core;

abstract class MiddleWare{
    public const MIDDLEWARE_RULE_IS_MUST = "MUST";
    public const MIDDLEWARE_RULE_OPTIONAL = "OPTIONAL";
    public const MIDDLEWARE_IS_MUST = "must";
    public const MIDDLEWARE_IS_OPTIONAL = "optional";
    private string $middleware_type;

    public function __construct($middleware_type){
        $this->middleware_type = $middleware_type;
    }

    public abstract function Rules(): array;

    public function load(){
        $validator = True;
        foreach($this->Rules() as $middleware_rule => $middleware_rule_type){
            if ($middleware_rule_type === self::MIDDLEWARE_RULE_IS_MUST)
                $validator = call_user_func([$this, $middleware_rule]);
            else
                call_user_func([$this, $middleware_rule]);
        }
        return $validator;
    }

    public function setMiddleWareType($type){
        $this->middleware_type = $type;
    }

    public function getMiddleWareType(): string{
        return $this->middleware_type;
    }
}
